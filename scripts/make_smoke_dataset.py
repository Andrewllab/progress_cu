"""Create synthetic plumbing-test data. This is not a learning benchmark."""
import json
from pathlib import Path

import h5py
import numpy as np


if __name__ == '__main__':
    destination = Path('.cache/smoke/data/synthetic.hdf5')
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    with h5py.File(destination, 'x') as file:
        data = file.create_group('data')
        data.attrs['env_args'] = json.dumps(dict(env_name='Lift', type=1, env_kwargs={}))
        for i in range(3):
            demo = data.create_group(f'demo_{i}')
            demo.attrs['num_samples'] = 48
            demo.create_dataset('actions', data=rng.normal(size=(48, 7)).astype('float32'))
            demo.create_dataset('rewards', data=np.zeros(48, dtype='float32'))
            demo.create_dataset('dones', data=np.zeros(48, dtype='int64'))
            obs = demo.create_group('obs')
            for key in ['agentview_image', 'robot0_eye_in_hand_image']:
                obs.create_dataset(key, data=rng.integers(0, 256, size=(48, 84, 84, 3), dtype='uint8'))
            for key, dimension in [('robot0_eef_pos', 3), ('robot0_eef_quat', 4), ('robot0_gripper_qpos', 2)]:
                obs.create_dataset(key, data=rng.normal(size=(48, dimension)).astype('float32'))
            next_obs = demo.create_group('next_obs')
            for key in obs:
                array = obs[key][:]
                next_obs.create_dataset(key, data=np.concatenate([array[1:], array[-1:]]))
        file.create_dataset('mask/train', data=np.array([b'demo_0', b'demo_1']))
        file.create_dataset('mask/valid', data=np.array([b'demo_2']))
    print(destination)
