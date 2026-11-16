import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer

def test():
    import numpy as np
    buff = ReplayBuffer.create_empty_numpy()
    buff.add_episode({
        'obs': np.zeros((100,10), dtype=np.float16)
    })
    buff.add_episode({
        'obs': np.ones((50,10)),
        'action': np.ones((50,2))
    })
    # buff.rechunk(256)
    obs = buff.get_episode(0)

    import numpy as np
    buff = ReplayBuffer.create_empty_zarr()
    buff.add_episode({
        'obs': np.zeros((100,10), dtype=np.float16)
    })
    buff.add_episode({
        'obs': np.ones((50,10)),
        'action': np.ones((50,2))
    })
    obs = buff.get_episode(0)
    buff.set_chunks({
        'obs': (100,10),
        'action': (100,2)
    })
