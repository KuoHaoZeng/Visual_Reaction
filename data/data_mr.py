import random, time, os, sys, math, json
sys.path.append('ai2thor')
from ai2thor.controller import Controller
import numpy as np
from threading import Thread
from multiprocessing import Process, Queue
try:
    from queue import Empty
except ImportError:
    from Queue import Empty
import torch

PATH = None

class drone_explorer(Thread):
    def __init__(self, pid, cfg, queue, mode='train'):
        super(drone_explorer, self).__init__()

        ### thread setting
        self.pid = pid
        set_random_seed(pid + cfg.framework.seed)
        self.cfg = cfg
        self.queue = queue
        self.iters = 0
        self.reset = False
        self.kill = False

        ### data
        self.objects = json.load(open(cfg.object_dir))
        if mode == 'train':
            self.trajectory = json.load(open(cfg.train.meta))
            self.object_random = cfg.train.random_object
        elif mode == 'val':
            self.trajectory = json.load(open(cfg.val.meta))
            self.object_random = cfg.val.random_object
        elif mode == 'test':
            self.trajectory = json.load(open(cfg.test.meta))
            self.object_random = cfg.test.random_object
        else:
            raise NotImplementedError

        self.trajectory_idx = np.arange(len(self.trajectory['scene']))
        np.random.shuffle(self.trajectory_idx)

        if pid == 0:
            for epoch in range(self.cfg.train.num_epoch):
                for ii in range(len(self.trajectory_idx)):
                    self.queue.put(self.trajectory_idx[ii])
        self.force, self.angle_x, self.angle_y = 0, 0, 0
        self.event = None

        ### THOR setting
        self.port = cfg.thor.port + pid
        self.x_display = '0.{}'.format(pid % cfg.framework.num_gpu + cfg.thor.x_display_offset)
        self.restart_cond = cfg.thor.restart_cond
        self.controller = None

        ### try it
        self.restart()

    def restart(self):
        ### reset the unity to avoid some latency issue ...
        if isinstance(self.controller, type(None)):
            self.controller = Controller(local_executable_path=PATH,
                                         scene="FloorPlan201_physics",
                                         x_display=self.x_display,
                                         agentMode='drone',
                                         fieldOfView=60,
                                         port=self.port)
        else:
            self.controller.stop()
            self.controller = Controller(local_executable_path=PATH,
                                         scene="FloorPlan201_physics",
                                         x_display=self.x_display,
                                         agentMode='drone',
                                         fieldOfView=60,
                                         port=self.port)
        _ = self.controller.reset(self.trajectory['scene'][0])
        _ = self.controller.step(dict(action='ChangeAutoResetTimeScale', timeScale=self.cfg.thor.time_scale))
        _ = self.controller.step(dict(action='ChangeFixedDeltaTime', fixedDeltaTime=self.cfg.thor.delta_time))
        print('Thread:'+str(self.pid)+' ['+str(self.iters+1)+']: restart finish')

    def run(self):
        while not self.kill:
            ### check if it needs to reset or the agent is still exploring
            if self.reset and not self.queue.empty():
                ### check if it reaches the restart condition
                if self.iters % self.restart_cond == 0 and self.iters > 1:
                    self.restart()

                ### get meta data of the trajectory
                try:
                    tidx = self.queue.get(timeout=1)
                except Empty:
                    if self.kill:
                        break
                    else:
                        self.good_to_go = False
                        self.reset = False
                        continue

                ### have a loop to ensure the trajectory has a good start
                self.good_to_go = False
                while not self.good_to_go:
                    scene = self.trajectory['scene'][tidx]
                    object_name = self.trajectory['object'][tidx]
                    mass = self.objects[object_name][1]
                    drone_position = self.trajectory['drone_position'][tidx]
                    launcher_position = self.trajectory['launcher_position'][tidx]
                    force = self.trajectory['force'][tidx]
                    angle_y = self.trajectory['angle_y'][tidx]
                    angle_x = self.trajectory['angle_x'][tidx]
                    if object_name == "Glassbottle":
                        object_name = "Bottle"

                    # make sure the value is .2f
                    drone_position['x'] = np.round(drone_position['x'], 2)
                    drone_position['y'] = np.round(1.5, 2)
                    drone_position['z'] = np.round(drone_position['z'], 2)
                    launcher_position['x'] = np.round(launcher_position['x'], 2)
                    launcher_position['y'] = np.round(launcher_position['y'], 2)
                    launcher_position['z'] = np.round(launcher_position['z'], 2)
                    force = np.round(force, 2)
                    angle_y = np.round(angle_y, 2)
                    angle_x = np.round(angle_x, 2)

                    ### set THOR
                    event = self.controller.reset(scene)
                    event = self.controller.step(dict(action='SpawnDroneLauncher', position=launcher_position))
                    event = self.controller.step(dict(action='FlyAssignStart',
                                                      position=drone_position,
                                                      x=launcher_position['x'],
                                                      y=launcher_position['y'],
                                                      z=launcher_position['z']))
                    event = self.controller.step(dict(action='Rotate', rotation=dict(x=0, y=0, z=0)))
                    event = self.controller.step(dict(action='ChangeAutoResetTimeScale',
                                                      timeScale=self.cfg.thor.time_scale))
                    event = self.controller.step(dict(action='ChangeFixedDeltaTime',
                                                      fixedDeltaTime=self.cfg.thor.delta_time))
                    if "noise_sigma" in self.cfg.agent:
                        event = self.controller.step(dict(action='ChangeDronePositionRandomNoiseSigma',
                                                          dronePositionRandomNoiseSigma=self.cfg.agent.noise_sigma))

                    ### prepare to launch the object
                    position = event.metadata['agent']['position']
                    event = self.controller.step(dict(action = 'LaunchDroneObject',
                                                      moveMagnitude = force,
                                                      x = angle_x,
                                                      y = angle_y,
                                                      z = -1,
                                                      objectName=object_name,
                                                      objectRandom=self.object_random))

                    if np.round(event.metadata['currentTime'], 2) == 0.00:
                        event = self.controller.step(dict(action='Pass'))

                    if np.round(event.metadata['currentTime'], 2) == 0.02:
                        self.good_to_go = True

                ### some record
                self.angle_x = angle_x
                self.angle_y = angle_y
                self.force = force
                self.object_name = object_name
                self.mass = mass
                self.event = event
                self.tidx = tidx

                ### thread setting
                self.reset = False
                self.iters += 1

            ### otherwise, just go to sleep
            else:
                time.sleep(0.1)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
