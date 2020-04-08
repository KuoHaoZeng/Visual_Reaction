import torch
import torch.nn as nn

dist = torch.distributions.Normal

def mpc_func(obs, agent_pos, model, ND, CD, horizon=20, weight_horizon=None,
             n_sample=10, actor=None, forecaster=None, gpu_id=0, action_scale=0.01):
    obs = obs.cuda(gpu_id)
    agent_pos = agent_pos.cuda(gpu_id)
    actions_horizon, agent_pos_horizon = agent_pos.clone(), agent_pos.clone()
    agent_pos_horizon = agent_pos_horizon.repeat(n_sample, 1)
    distances_horizon = torch.zeros(n_sample).cuda(gpu_id)
    for i in range(min(horizon, len(obs))):
        obs_i = obs[i,:]

        ## sample an action
        if isinstance(actor,type(None)):
            a = torch.rand(n_sample, ND).unsqueeze(1).cuda(gpu_id) * (CD - 1)
            a = (a - (CD - 1) // 2) * action_scale
        else:
            actor.batch_size = 1
            vel = model.vel.clone()
            if vel.shape[0] != n_sample:
                vel = vel.repeat(n_sample, 1)
            inp = torch.cat((obs_i.repeat(n_sample,1), agent_pos_horizon, vel), dim=1)
            _, action_dist_mu, action_dist_sigma = actor(inp, forecaster.representation)
            a = dist(action_dist_mu, action_dist_sigma).sample()
            a = torch.clamp(a, -(CD - 1) * action_scale / 2, (CD - 1) * action_scale / 2)
            a = a.unsqueeze(1)

        ## use dynamics model to forward a time step
        new_agent_pos = agent_pos_horizon.unsqueeze(1)
        inp_s = torch.cat((new_agent_pos, a), dim=1)
        model.batch_size = n_sample
        # forward
        new_agent_pos = model(inp_s)

        ## objective for MPC
        obs_i = obs_i.repeat(n_sample,1)
        distance_s = (obs_i - new_agent_pos).abs().sum(dim=1)
        if not isinstance(weight_horizon, type(None)):
            distance_s *= weight_horizon[i]

        if i == 0:
            actions_horizon = a.clone()
        distances_horizon += distance_s.clone()
        agent_pos_horizon = new_agent_pos.clone()

    best_id = distances_horizon.argmin()
    best_distance = distances_horizon.min().clone()
    best_a = actions_horizon[best_id].clone()
    return best_distance, best_a
