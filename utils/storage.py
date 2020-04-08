import numpy as np
from threading import Lock
lock = Lock()

class Memory():
    def __init__(self, size, batch_size):
        ### size: max size of the memory
        ### batch_size: size of a batch of data
        self.size = size
        self.batch_size = batch_size
        self.dict = {}

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        if len(self.dict.keys()) > 0:
            return len(self.dict[list(self.dict.keys())[0]])
        else:
            return len(self.dict.keys())

    def is_full(self):
        if len(self) >= self.size:
            return True
        else:
            return False

    def is_enough(self):
        if len(self) >= self.batch_size:
            return True
        else:
            return False

    def add_to(self):
        raise NotImplementedError

class Memory_WF(Memory):
    def __init__(self, size, batch_size):
        super(Memory_WF, self).__init__(size, batch_size)
        self.remove_idx = None

    def __getitem__(self, keys):
        lock.acquire()
        idx = np.random.choice(len(self), self.batch_size)
        output = {}
        for k in keys:
            output[k] = self.dict[k][idx]
        self.clean(idx)
        lock.release()
        return output

    def add_to(self, key, data):
        #data = np.array([data])
        if key not in self.dict:
            self.dict[key] = data
        else:
            if len(data.shape) > 1:
                self.dict[key] = np.vstack([self.dict[key], data])
            else:
                self.dict[key] = np.hstack([self.dict[key], data])
        if len(self) > self.size:
            self.clean()

    def clean(self, idx=None):
        if isinstance(idx, type(None)):
            if len(self) > self.size:
                self.set_remove_idx()
                for k, v in self.dict.items():
                    self.dict[k] = np.delete(self.dict[k], self.remove_idx, axis=0)
        else:
            for k, v in self.dict.items():
                self.dict[k] = np.delete(self.dict[k], idx, axis=0)

    def set_remove_idx(self):
        self.remove_idx = np.array(list(range(self.size, len(self))))

class Memory_Seq(Memory_WF):
    def __init__(self, size, batch_size):
        super(Memory_Seq, self).__init__(size, batch_size)

    def __getitem__(self, keys):
        lock.acquire()
        idx = np.arange(self.batch_size)
        output = {}
        for k in keys:
            output[k] = self.dict[k][idx]
        self.clean(idx)
        lock.release()
        return output

class Storage():
    def __init__(self):
        self.origin = None
        self.dict = {}

    def __len__(self):
        if len(self.dict.keys()) > 0:
            return len(self.dict[list(self.dict.keys())[0]])
        else:
            return len(self.dict.keys())

    def add_to(self, key, data):
        data = np.array([data])
        if key not in self.dict:
            self.dict[key] = data
        else:
            if len(data.shape) > 1:
                self.dict[key] = np.vstack([self.dict[key], data])
            else:
                self.dict[key] = np.hstack([self.dict[key], data])

def record(storage, event, action_acc):
    # get position of drone and flying object from unity first
    agent_pos = np.array([ele for ele in event.metadata['agent']['position'].values()])
    ball_pos = np.array([ele for ele in event.metadata['objects'][0]['position'].values()])

    # check if it's the first time for storage
    if len(storage) == 0:
        storage.origin = np.array(agent_pos)

    # keep in mind that we use relative position
    agent_pos -= storage.origin
    ball_pos -= storage.origin
    storage.add_to('agent_pos', agent_pos)
    storage.add_to('ball_pos', ball_pos)

    # store velocity and acceleration
    if 'ball_vel' in storage.dict.keys():
        storage.add_to('ball_vel', storage.dict['ball_pos'][-1] - storage.dict['ball_pos'][-2])
        storage.add_to('ball_acc', storage.dict['ball_vel'][-1] - storage.dict['ball_vel'][-2])
        storage.add_to('agent_vel', storage.dict['agent_pos'][-1] - storage.dict['agent_pos'][-2])
    else:
        launcher_pos = np.array([ele for ele in event.metadata['agent']['LauncherPosition'].values()])
        launcher_pos -= storage.origin
        storage.add_to('ball_vel', ball_pos - launcher_pos)
        storage.add_to('ball_acc', [0, 0, 0])
        storage.add_to('agent_vel', [0, 0, 0])

    # other stuff
    storage.add_to('time', np.round(event.metadata['droneCurrentTime'], 2))
    storage.add_to('num_SimObj_hits', event.metadata['objects'][0]['numSimObjHits'])
    storage.add_to('num_Floor_hits', event.metadata['objects'][0]['numFloorHits'])
    storage.add_to('num_Structure_hits', event.metadata['objects'][0]['numStructureHits'])
    storage.add_to('visible', event.metadata['objects'][0]['visible'])
    storage.add_to('action', event.metadata['lastAction'])
    storage.add_to('action_valid', event.metadata['lastActionSuccess'])
    storage.add_to('success', event.metadata['objects'][0]['isCaught'])
    storage.add_to('terminal', False)

    # store frames
    frame = np.transpose(event.frame, (2, 0, 1))
    if 'frame' in storage.dict.keys():
        if len(storage.dict['frame']) < 2:
            frame_t_0 = np.expand_dims(storage.dict['frame'][-1, -1], 0)
        else:
            frame_t_0 = np.expand_dims(storage.dict['frame'][-2, -1], 0)
        frame_t_1 = np.expand_dims(storage.dict['frame'][-1, -1], 0)
        frame_t_2 = np.expand_dims(frame, 0)
        frame = np.vstack((frame_t_0, frame_t_1, frame_t_2))
        storage.add_to('frame', frame)
    else:
        storage.add_to('frame', np.repeat(np.expand_dims(frame, 0), 3, axis=0))

    # store the entire event
    storage.add_to('event', event)

    # store the executed action
    storage.add_to('agent_acc', action_acc)
    return storage, agent_pos, ball_pos

def record_agnet(storage, agent_pos_est, agent_vel_est, agent_acc_est, frame, agent_angle_y, agent_angle_xz, ball_pos_est, ball_vel_est, ball_acc_est):
    storage.add_to('agent_pos_est', agent_pos_est)
    storage.add_to('agent_vel_est', agent_vel_est)
    storage.add_to('agent_acc_est', agent_acc_est)
    storage.add_to('agent_angle_y', agent_angle_y)
    storage.add_to('agent_angle_xz', agent_angle_xz)
    storage.add_to('ball_pos_est', ball_pos_est)
    storage.add_to('ball_vel_est', ball_vel_est)
    storage.add_to('ball_acc_est', ball_acc_est)

    # store observation
    storage.add_to('input_frames', frame)
    return storage

def record_meta(storage, event, explorer):
    storage.add_to('tidx', explorer.tidx)
    storage.add_to('angle_x', explorer.angle_x)
    storage.add_to('angle_y', explorer.angle_y)
    storage.add_to('force', explorer.force)
    storage.add_to('object_name', explorer.object_name)
    storage.add_to('mass', explorer.mass)
    storage.add_to('initial_agnet_pos', [ele for ele in event.metadata['agent']['position'].values()])
    storage.add_to('launcher_pos', [ele for ele in event.metadata['agent']['LauncherPosition'].values()])
    return storage

def get_memory_data(memory_buffer):
    data_key = ['ball_pos',
                'ball_pos_next',
                'ball_vel',
                'ball_acc',
                'input_frames',
                'agent_pos_est',
                'agent_vel_est',
                'agent_acc_est',
                'agent_pos',
                'agent_vel',
                'agent_acc',
                'agent_angle_y',
                'agent_angle_xz',
                'success',
                'terminal']
    memory_data = memory_buffer[data_key]
    return memory_data
