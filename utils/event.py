import torch, sys
import numpy as np

def get_distance(event_a, event_b):
    p_a = event_a.metadata['objects'][0]['position']
    p_b = event_b.metadata['objects'][0]['position']
    distance = (p_a['y'] - p_b['y'])**2 +\
               (p_a['x'] - p_b['x'])**2 +\
               (p_a['z'] - p_b['z'])**2
    distance = distance**0.5
    return distance

def get_deltime(event_a, event_b):
    return np.round(abs(event_a.metadata['droneCurrentTime'] - event_b.metadata['droneCurrentTime']), 2)

def check_with_time_difference(event_a, event_b):
    del_time = get_deltime(event_a, event_b)
    if del_time > 0.02:
        tag = 'fail'
    elif del_time < 0.02:
        tag = 'continue'
    else:
        tag = 'fine'
    return tag

def get_relative_pos(event):
    p_a = event.metadata['objects'][0]['position']
    p_b = event.metadata['agent']['position']
    pos = [p_a['x'] - p_b['x'], p_a['y'] - p_b['y'], p_a['z'] - p_b['z']]
    return pos

def get_camera_pos(p_a, p_b):
    rel_pos = p_a - p_b
    xz = (rel_pos[:,2] > 0) * (torch.atan(rel_pos[:,0]/rel_pos[:,2])*180/np.pi) +\
         (rel_pos[:,2] < 0) * (rel_pos[:,0] > 0) * (torch.atan(rel_pos[:,0]/rel_pos[:,2])*180/np.pi - 180) +\
         (rel_pos[:,2] < 0) * (rel_pos[:,0] < 0) * (torch.atan(rel_pos[:,0]/rel_pos[:,2])*180/np.pi + 180 )
    yz = (rel_pos[:,2] > 0) * (torch.atan(rel_pos[:,1]/rel_pos[:,2])*180/np.pi) +\
         (rel_pos[:,2] < 0) * (-torch.atan(rel_pos[:,1]/rel_pos[:,2])*180/np.pi)

    return yz, xz

def get_camera_pos_numpy(event):
    rel_pos = get_relative_pos(event)
    if rel_pos[2] > 0:
        xz = np.arctan(rel_pos[0]/rel_pos[2])*180/np.pi
        yz = np.arctan(rel_pos[1]/rel_pos[2])*180/np.pi
    else:
        if rel_pos[0] > 0:
            xz = np.arctan(rel_pos[0]/rel_pos[2])*180/np.pi - 180
        else:
            xz = np.arctan(rel_pos[0]/rel_pos[2])*180/np.pi + 180
        yz = -np.arctan(rel_pos[1]/rel_pos[2])*180/np.pi
    return yz, xz
