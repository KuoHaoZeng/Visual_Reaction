import time, os, argparse, jsonlines, progressbar
import numpy as np
from threading import Thread, Barrier, Lock
from multiprocessing import Queue

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from network.network import *
from utils.storage import *
from utils.event import *
from utils.utils import *
from utils.config import Config
from mpc.mpc import mpc_func
from data.data_mr import *


class trainer_Online_VA(Thread):
    def __init__(self, t_id, cfg, log_queue, data_queue, memory_buffer, mode, explorer,
                 share_forecaster, share_actor, optimizer_f, optimizer_a, scheduler_f):
        super(trainer_Online_VA, self).__init__()

        ### somethings
        set_random_seed(t_id + cfg.framework.seed)
        self.t_id = t_id
        self.cfg = cfg
        self.mode = mode
        self.data_queue = data_queue
        self.memory_buffer = memory_buffer
        self.gpu_id = t_id % cfg.framework.num_gpu + cfg.thor.x_display_offset
        self.train_iter = 0
        self.num_traj = 0
        self.total_traj = len(explorer.trajectory['scene'])

        ### logging
        self.log_keys = ['ball_pos', 'ball_vel', 'ball_acc', 'agent_pos', 'agent_acc',
                         'agent_pos_est', 'agent_vel_est', 'agent_acc_est', 'agent_angle_y', 'agent_angle_xz',
                         'success', 'num_SimObj_hits', 'num_Floor_hits', 'num_Structure_hits', 'tidx',
                         'angle_x', 'angle_y', 'force', 'object_name', 'mass', 'initial_agnet_pos', 'launcher_pos',
                         'ball_pos_est', 'ball_vel_est', 'ball_acc_est']
        self.log_queue = log_queue
        if self.t_id == 0:
            self.log_dir = "{}/{}_{}".format(cfg.base_dir, mode, cfg.log_dir)
            self.log_writer = open(self.log_dir, 'wb')
            if mode == "train":
                widgets = ['Training phase [', progressbar.SimpleProgress(), '] [', progressbar.Percentage(), '] ',
                           progressbar.Bar(marker='█'), ' (', progressbar.Timer(), ' ', progressbar.ETA(), ') ', ]
            else:
                widgets = ['Evaluation phase [', progressbar.SimpleProgress(), '] [', progressbar.Percentage(), '] ',
                           progressbar.Bar(marker='█'), ' (', progressbar.Timer(), ' ', progressbar.ETA(), ') ', ]
            if mode == "train":
                max_value = self.total_traj * cfg.train.num_epoch
            else:
                max_value = self.total_traj
            self.bar = progressbar.ProgressBar(max_value=max_value, widgets=widgets, term_width=100)
        else:
            self.log_dir, self.writer = None, None

        ### define loss functions
        self.criterionL1 = nn.L1Loss()

        ### explorer
        self.explorer = explorer

        ### forecaster
        self.share_forecaster = share_forecaster
        self.forecaster = forecaster_protocols[cfg.agent.protocol](1,
                                                                   cfg.agent.ND,
                                                                   cfg.agent.hidden_size).cuda(self.gpu_id)
        if mode == "train":
            self.forecaster.train()
        else:
            self.forecaster.eval()
        self.optimizer_f, self.scheduler_f = optimizer_f, scheduler_f

        ### model
        self.model = model_protocols[cfg.agent.protocol_model](1,
                                                               cfg.agent.ND,
                                                               cfg.agent.hidden_size,
                                                               cfg.thor.delta_time,
                                                               True,
                                                               self.gpu_id).cuda(self.gpu_id)
        self.model.gpu_id = self.gpu_id

        ### actor
        self.share_actor = share_actor
        if not isinstance(self.cfg.agent.protocol_actor, type(None)):
            self.actor = actor_protocols[cfg.agent.protocol_actor](1,
                                                                   cfg.agent.ND,
                                                                   cfg.agent.hidden_size,
                                                                   cfg.agent.CD).cuda(self.gpu_id)
            self.dist = torch.distributions.Normal
            if mode == "train":
                self.actor.train()
            else:
                self.actor.eval()
                self.forecaster.eval()
        else:
            self.actor, self.dist = None, None
        self.optimizer_a = optimizer_a

        if not isinstance(self.cfg.checkpoint_file, type(None)) and len(os.listdir(self.cfg.checkpoint_dir)) > 0:
            checkpoint = "{}/{:07d}".format(self.cfg.checkpoint_dir, self.cfg.checkpoint_file)
            self.load_checkpoints(checkpoint)

    def save_checkpoints(self):
        dir = "{}/{:07d}".format(self.cfg.checkpoint_dir, self.explorer.iters * self.cfg.framework.num_thread)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        sd = {}
        if not isinstance(self.forecaster, type(None)):
            sd['forecaster'] = self.share_forecaster.module.state_dict()
        if not isinstance(self.actor, type(None)):
            sd['actor'] = self.share_actor.module.state_dict()

        checkpoint_dir = "{}/model.pt".format(dir)
        torch.save(sd, checkpoint_dir)

    def load_checkpoints(self, checkpoint_dir):
        sd = torch.load("{}/model.pt".format(checkpoint_dir), map_location=torch.device('cpu'))
        self.train_iter = 0

        if not isinstance(self.share_forecaster, type(None)):
            if 'forecaster' in sd.keys():
                self.share_forecaster.module.load_state_dict(sd['forecaster'])
        if not isinstance(self.share_actor, type(None)):
            if 'actor' in sd.keys():
                self.share_actor.module.load_state_dict(sd['actor'])

    def run(self):
        self.explorer.start()

        while not self.data_queue.empty():
            ### Sync with the shared model
            if not isinstance(self.forecaster, type(None)):
                self.forecaster.load_state_dict(self.share_forecaster.module.state_dict())
            if not isinstance(self.actor, type(None)):
                self.actor.load_state_dict(self.share_actor.module.state_dict())

            if self.t_id == 0:
                if self.mode == 'train':
                    if self.explorer.iters > 1:
                        if (self.explorer.iters * self.cfg.framework.num_thread) % self.cfg.train.save_iter == 0:
                            self.save_checkpoints()

            if len(self.memory_buffer) < self.cfg.framework.buffer_size or self.mode != 'train':

                storage = Storage()

                ### reset the initial velocity
                self.model.reset_vel()

                ### Reset the environment
                self.explorer.reset = True

                ## Exploration
                while self.explorer.reset:
                    time.sleep(0.01)

                if not self.explorer.good_to_go:
                    if self.data_queue.empty():
                        break
                    else:
                        continue

                ## first
                event = self.explorer.event
                storage, agent_pos, ball_pos = record(storage, event, [0, 0, 0])
                agent_angle_y, agent_angle_xz = get_camera_pos_numpy(event)
                storage = record_agnet(storage, agent_pos, [0, 0, 0], [0, 0, 0], storage.dict['frame'][-1],
                                       agent_angle_y, agent_angle_xz, ball_pos, [0, 0, 0], [0, 0, 0])
                storage = record_meta(storage, event, self.explorer)

                ## register what's the first time step the data is ready to used
                storage.add_to('first', 3)

                while len(storage) < self.cfg.thor.max_time:
                    ## wait till the first three frames are collected
                    if len(storage) < 3:
                        event = self.explorer.controller.step(dict(action='Pass'))
                        tag = check_with_time_difference(event, storage.dict['event'][-1])
                        if tag == 'fail':
                            break
                        elif tag == 'continue':
                            continue
                        else:
                            pass
                        storage, agent_pos, ball_pos = record(storage, event, [0, 0, 0])
                        agent_angle_y, agent_angle_xz = get_camera_pos_numpy(event)
                        storage = record_agnet(storage, agent_pos, [0, 0, 0], [0, 0, 0], storage.dict['frame'][-1],
                                               agent_angle_y, agent_angle_xz, ball_pos, storage.dict['ball_vel'][-1],
                                               storage.dict['ball_acc'][-1])
                    ## or to policy evaluation
                    else:
                        # Data processing
                        input_frames = storage.dict['frame'][-1]

                        # check if it's the first time policy evaluation
                        if len(storage) == 3:
                            # expand the first dimension for batch size
                            agent_pos_est = torch.Tensor(np.array(agent_pos)).cuda(self.gpu_id)
                            agent_vel_est = torch.Tensor(np.array([0, 0, 0])).cuda(self.gpu_id)
                            agent_acc_est = torch.Tensor(np.array([0, 0, 0])).cuda(self.gpu_id)
                            agent_pos_est = agent_pos_est.unsqueeze(0)
                            agent_vel_est = agent_vel_est.unsqueeze(0)
                            agent_acc_est = agent_acc_est.unsqueeze(0)
                            ball_pos_est = torch.Tensor(np.array(ball_pos)).cuda(self.gpu_id)
                            ball_vel_est = torch.Tensor(np.array(storage.dict['ball_vel'][-1])).cuda(self.gpu_id)
                            ball_acc_est = torch.Tensor(np.array(storage.dict['ball_acc'][-1])).cuda(self.gpu_id)
                            ball_pos_est = ball_pos_est.unsqueeze(0)
                            ball_vel_est = ball_vel_est.unsqueeze(0)
                            ball_acc_est = ball_acc_est.unsqueeze(0)
                            agent_angle_y = torch.Tensor([agent_angle_y]).cuda(self.gpu_id)
                            agent_angle_xz = torch.Tensor([agent_angle_xz]).cuda(self.gpu_id)

                        storage = record_agnet(storage,
                                               agent_pos_est.squeeze(0).cpu().numpy(),
                                               agent_vel_est.squeeze(0).cpu().numpy(),
                                               agent_acc_est.squeeze(0).cpu().numpy(),
                                               input_frames,
                                               agent_angle_y.item(),
                                               agent_angle_xz.item(),
                                               ball_pos_est.squeeze(0).cpu().numpy(),
                                               ball_vel_est.squeeze(0).cpu().numpy(),
                                               ball_acc_est.squeeze(0).cpu().numpy())

                        ## Planner, Input: first three images
                        # forward forecaster to obtain the anticipated trajectory
                        input_frames = torch.Tensor(input_frames).cuda(self.gpu_id)
                        ball_pos_est, ball_vel_est, ball_acc_est, _ = self.forecaster(input_frames,
                                                                                       agent_pos_est,
                                                                                       agent_vel_est,
                                                                                       agent_acc_est,
                                                                                       agent_angle_y,
                                                                                       agent_angle_xz)
                        # detach the graph
                        ball_pos_est, ball_vel_est, ball_acc_est = ball_pos_est.detach(), ball_vel_est.detach(), ball_acc_est.detach()

                        # obtain the predicted vel and acc by the highest prob.
                        p_pred = ball_pos_est.clone()
                        v_pred = ball_vel_est.clone()
                        a_pred = ball_acc_est.clone()

                        # anticipate trajectory
                        data_forecast = self.forecaster.forecast(p_pred, v_pred, a_pred, self.cfg.agent.Horizon)
                        horizon = self.cfg.agent.Horizon

                        # minus 0.44 to let agent fly below the target trajectory
                        data_forecast[:, 1] = data_forecast[:, 1] - self.cfg.agent.height_difference

                        # clone the dynamics model for MPC usage
                        model_to_mpc = model_protocols[self.cfg.agent.protocol_model](1,
                                                                                      self.cfg.agent.ND,
                                                                                      self.cfg.agent.hidden_size,
                                                                                      self.cfg.thor.delta_time,
                                                                                      True,
                                                                                      self.gpu_id).cuda(self.gpu_id)
                        model_to_mpc.vel = self.model.vel.clone()

                        # do MPC
                        # weight_horizon = [0.95**ele for ele in range(self.cfg.agent.Horizon)]
                        weight_horizon = None
                        dis, action = mpc_func(data_forecast,
                                               agent_pos_est,
                                               model_to_mpc,
                                               self.cfg.agent.ND,
                                               self.cfg.agent.CD,
                                               horizon,
                                               weight_horizon,
                                               self.cfg.agent.NSample,
                                               self.actor,
                                               self.forecaster,
                                               self.gpu_id,
                                               self.cfg.agent.DScale)

                        # detach the graph
                        dis, action = dis.detach(), action.detach()

                        # forward again by the selected action
                        inp_s = torch.cat((agent_pos_est.unsqueeze(0), action.unsqueeze(1)), dim=1)
                        agent_pos_est = self.model(inp_s)
                        agent_vel_est = self.model.vel.detach().clone()
                        agent_acc_est = action.clone()

                        # detach the graph
                        agent_pos_est = agent_pos_est.detach()

                        # compute camera pose
                        ball_pos_est_for_camera = data_forecast[0, :].unsqueeze(0)
                        ball_pos_est_for_camera[:, 1] += self.cfg.agent.height_difference
                        agent_pos_est_for_camera = agent_pos_est.clone()
                        if self.cfg.agent.gt_camera_pose:
                            agent_angle_y, agent_angle_xz = get_camera_pos_numpy(event)
                            agent_angle_y = torch.Tensor([agent_angle_y]).cuda(self.gpu_id)
                            agent_angle_xz = torch.Tensor([agent_angle_xz]).cuda(self.gpu_id)
                        else:
                            agent_angle_y, agent_angle_xz = get_camera_pos(ball_pos_est_for_camera,
                                                                           agent_pos_est_for_camera)
                        agent_angle_y = torch.round(
                            agent_angle_y.clone() * (1 / self.cfg.agent.PScale)) * self.cfg.agent.PScale
                        agent_angle_xz = torch.round(
                            agent_angle_xz.clone() * (1 / self.cfg.agent.PScale)) * self.cfg.agent.PScale

                        # send action commands to THOR
                        event = self.explorer.controller.step(dict(action='FlyTo',
                                                                   x=action[0, 0].item(),
                                                                   y=action[0, 1].item(),
                                                                   z=action[0, 2].item(),
                                                                   horizon=-agent_angle_y.item(),
                                                                   rotation=agent_angle_xz.item()))

                        # check time difference
                        tag = check_with_time_difference(event, storage.dict['event'][-1])
                        while tag == 'continue':
                            event = self.explorer.controller.step(dict(action='Pass'))
                            tag = check_with_time_difference(event, storage.dict['event'][-1])
                        if tag == 'fail':
                            break
                        else:
                            pass

                        # record data
                        storage, agent_pos, ball_pos = record(storage,
                                                              event,
                                                              [action[0, 0].item(), action[0, 1].item(),
                                                               action[0, 2].item()])

                        # check if the trajectory ends
                        if event.metadata['objects'][0]['isCaught']:
                            break
                        elif get_distance(storage.dict['event'][-1], storage.dict['event'][-2]) < 0.00001:
                            if ((storage.dict['ball_vel'][-1] ** 2).sum() ** 0.5) < 0.00001:
                                if ((storage.dict['ball_acc'][-1] ** 2).sum() ** 0.5) < 0.00001:
                                    break

                if tag != 'fail':
                    lock.acquire()
                    start_time = storage.dict['first'][0]
                    log_meta = {}
                    storage.dict['terminal'][-1] = True
                    for k, v in storage.dict.items():
                        if k != 'event' and k != 'frames':
                            if self.mode == 'train':
                                if k == 'ball_pos':
                                    self.memory_buffer.add_to(k, v[start_time - 1: -1])
                                    self.memory_buffer.add_to('ball_pos_next', v[start_time:])
                                else:
                                    self.memory_buffer.add_to(k, v[start_time:])
                        if k in self.log_keys:
                            log_meta[k] = v.tolist()
                    lock.release()
                    self.log_queue.put(log_meta)

                    if self.t_id == 0:
                        log_meta = []
                        while not self.log_queue.empty():
                            log_meta.append(self.log_queue.get())
                        if len(log_meta) > 0:
                            self.num_traj = write_log(self.log_writer, log_meta, self.num_traj, self.bar)
                else:
                    self.data_queue.put(self.explorer.tidx)

            ### training
            if len(self.memory_buffer) >= self.cfg.framework.batch_size and self.mode == 'train' and self.t_id == 0:
                ## get data
                lock.acquire()
                memory_data = get_memory_data(self.memory_buffer)
                lock.release()

                # get ball data
                ball_pos = torch.Tensor(memory_data['ball_pos']).cuda(self.gpu_id)
                ball_pos_next = torch.Tensor(memory_data['ball_pos_next']).cuda(self.gpu_id)
                ball_vel = torch.Tensor(memory_data['ball_vel']).cuda(self.gpu_id)
                ball_acc = torch.Tensor(memory_data['ball_acc']).cuda(self.gpu_id)

                # get agent data
                agent_pos_est = torch.Tensor(memory_data['agent_pos_est']).cuda(self.gpu_id)
                agent_vel_est = torch.Tensor(memory_data['agent_vel_est']).cuda(self.gpu_id)
                agent_acc_est = torch.Tensor(memory_data['agent_acc_est']).cuda(self.gpu_id)
                agent_pos = torch.Tensor(memory_data['agent_pos']).cuda(self.gpu_id)
                agent_vel = torch.Tensor(memory_data['agent_vel']).cuda(self.gpu_id)
                agent_acc = torch.Tensor(memory_data['agent_acc']).cuda(self.gpu_id)
                agent_angle_y = torch.Tensor(memory_data['agent_angle_y']).cuda(self.gpu_id)
                agent_angle_xz = torch.Tensor(memory_data['agent_angle_xz']).cuda(self.gpu_id)

                # get frames
                input_frames = torch.Tensor(memory_data['input_frames']).cuda(self.gpu_id)
                success = torch.Tensor(memory_data['success']).cuda(self.gpu_id)
                terminal = torch.Tensor(memory_data['terminal']).cuda(self.gpu_id)

                if self.cfg.agent.forecaster.train:
                    ## Forecaster training
                    ball_pos_est, ball_vel_est, ball_acc_est, _ = self.share_forecaster(input_frames,
                                                                                        agent_pos_est,
                                                                                        agent_vel_est,
                                                                                        agent_acc_est,
                                                                                        agent_angle_y,
                                                                                        agent_angle_xz)

                    # get ground truth and compute sub-losses
                    gt_pos = ball_pos.clone()
                    loss_pos = self.criterionL1(ball_pos_est, gt_pos)

                    gt_vel = ball_vel.clone()
                    loss_vel = self.criterionL1(ball_vel_est, gt_vel)

                    gt_acc = ball_acc.clone()
                    loss_acc = self.criterionL1(ball_acc_est, gt_acc)

                    # compute loss
                    loss_forecaster = loss_pos + 0.1 * loss_vel + 0.1 * loss_acc

                    # backward and update
                    self.share_forecaster.zero_grad()
                    loss_forecaster.backward()
                    self.optimizer_f.step()
                    self.scheduler_f.step()

                if self.cfg.agent.actor.train and not isinstance(self.actor, type(None)):
                    ## actor training
                    ball_pos_next[:, 1] -= self.cfg.agent.height_difference
                    distance = ((ball_pos_next - agent_pos) ** 2).sum(1) ** 0.5
                    ball_pos_est, ball_vel_est, ball_acc_est, representation = self.share_forecaster(input_frames,
                                                                                                     agent_pos_est,
                                                                                                     agent_vel_est,
                                                                                                     agent_acc_est,
                                                                                                     agent_angle_y,
                                                                                                     agent_angle_xz)
                    # anticipate trajectory
                    data_forecast = ball_pos_est + ball_vel_est
                    data_forecast = torch.cat((data_forecast, agent_pos_est, agent_vel_est), dim=1).clone().detach()
                    representation = representation.clone().detach()
                    # compute loss
                    v_est, action_est_mu, action_est_sigma = self.share_actor(data_forecast, representation)
                    m = self.dist(action_est_mu, action_est_sigma)
                    log_prob = m.log_prob(agent_acc)

                    index = list(range(self.cfg.framework.batch_size - 1))
                    index.reverse()

                    loss_v, loss_a = 0., 0.
                    R = success[-1] - 0.01 * distance[-1]
                    for ii in index:
                        if terminal[ii]:
                            R = success[ii] - 0.01 * distance[ii]
                            continue

                        v = v_est[ii]
                        R = self.cfg.agent.actor.gamma * R + success[ii] - 0.01 * distance[ii]
                        advangtage = R.clone().detach() - v.clone().detach()
                        loss_v = loss_v + 0.5 * (R.clone().detach() - v).pow(2)

                        loss_a = loss_a - (advangtage * log_prob[ii]).mean()

                    loss_a /= self.cfg.framework.batch_size
                    loss_v /= self.cfg.framework.batch_size

                    loss_actor = (loss_a + 0.5 * loss_v - self.cfg.agent.actor.beta * m.entropy().mean())

                    # backward and update
                    self.share_actor.zero_grad()
                    loss_actor.backward()
                    clip_grad_norm_(self.share_actor.parameters(), 0.5)
                    self.optimizer_a.step()

                self.train_iter += 1

        # wait all threads finish
        if self.t_id == 0:
            log_meta = []
            while not self.log_queue.empty():
                log_meta.append(self.log_queue.get())
            if len(log_meta) > 0:
                self.num_traj = write_log(self.log_writer, log_meta, self.num_traj, self.bar)
            if self.mode == 'train':
                self.save_checkpoints()

class master_Online():
    def __init__(self, cfg, mode):
        ### allocate explorer
        data_queue = Queue()
        explorers = [drone_explorer(t_id, cfg, data_queue, mode) for t_id in range(cfg.framework.num_thread)]
        device_ids = [g_id for g_id in range(cfg.thor.x_display_offset, cfg.thor.x_display_offset + cfg.framework.num_gpu)]

        ### construct a forecaster and, if necessary, its optimizer as well as scheduler
        share_forecaster = forecaster_protocols[cfg.agent.protocol](
            cfg.framework.batch_size // cfg.framework.num_gpu,
            cfg.agent.ND,
            cfg.agent.hidden_size)
        share_forecaster = share_forecaster.cuda(cfg.thor.x_display_offset)
        share_forecaster = nn.DataParallel(share_forecaster, device_ids=device_ids)
        if mode == 'train':
            share_forecaster.train()
            share_forecaster.share_memory()
            optimizer_f = optim.SGD(share_forecaster.parameters(), lr=cfg.train.lr_f)
            scheduler_f = optim.lr_scheduler.MultiStepLR(optimizer_f,
                                                         milestones=cfg.train.lr_f_ms,
                                                         gamma=0.1)
        else:
            optimizer_f, scheduler_f = None, None

        ### construct an action sampler and, if necessary, its optimizer as well as scheduler
        if not isinstance(cfg.agent.protocol_actor, type(None)):
            share_actor = actor_protocols[cfg.agent.protocol_actor](
                cfg.framework.batch_size // cfg.framework.num_gpu,
                cfg.agent.ND,
                cfg.agent.hidden_size,
                cfg.agent.CD).cuda(cfg.thor.x_display_offset)
            share_actor = nn.DataParallel(share_actor, device_ids=device_ids)
            if cfg.agent.actor.train:
                optimizer_a = optim.Adam(share_actor.parameters(), lr=cfg.train.lr_a)
            else:
                optimizer_a = None
        else:
            share_actor, optimizer_a = None, None

        if isinstance(optimizer_a, type(None)):
            memory_buffer = Memory_WF(cfg.framework.buffer_size, cfg.framework.batch_size)
        else:
            memory_buffer = Memory_Seq(cfg.framework.buffer_size, cfg.framework.batch_size)

        ### prepare training/validation/testing jobs
        self.job = []
        log_queue = Queue()
        for t_id in range(cfg.framework.num_thread):
            self.job.append(protocols[cfg.agent.protocol](t_id, cfg, log_queue, data_queue,
                                                          memory_buffer, mode, explorers[t_id],
                                                          share_forecaster, share_actor,
                                                          optimizer_f, optimizer_a, scheduler_f))

    def run(self):
        for job in self.job:
            job.start()

        for job in self.job:
            job.join()


def get_configs():
    parser = argparse.ArgumentParser(description="Online mode training")
    parser.add_argument("--config", type=str, default="configs/pretrained_action_sampler_test.yaml")
    args = parser.parse_args()
    config = Config(args.config)
    return config


def main(cfg):
    if not os.path.isdir(cfg.base_dir):
        os.makedirs(cfg.base_dir)
    m = master_Online(cfg, cfg.mode)
    m.run()


if __name__ == '__main__':
    protocols = {'VA': trainer_Online_VA}
    forecaster_protocols = {'VA': NetForecaster}
    model_protocols = {'PM': DroneModel}
    actor_protocols = {'RL': NetPolicy}

    config = get_configs()
    barrier = Barrier(config.framework.num_thread)
    lock = Lock()
    main(config)
