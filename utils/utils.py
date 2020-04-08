import json
import sys
import random
import numpy as np
import torch

def print_progress(txt):
    print("{}".format(txt), end='\r', flush=True)
    sys.stdout.flush()

def write_log(log_writer, log_meta, num_traj, bar):
    assert len(log_meta) > 0
    for line in log_meta:
        out = json.dumps(line, ensure_ascii=False)
        log_writer.write(out.encode('utf-8'))
        log_writer.write('\n'.encode('utf-8'))
        num_traj += 1

    bar.update(num_traj)
    return num_traj

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
