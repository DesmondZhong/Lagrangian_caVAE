import os, sys
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import numpy as np
import gym
import myenv
from skimage import color
from utils import to_pickle, from_pickle


def sample_gym(seed=0, timesteps=10, trials=50,
              verbose=False, u=0.0, env_name='MyPendulum-v0'):
    
    gym_settings = locals()
    if verbose:
        print("Making a dataset of Pendulum observations.")
    env = gym.make(env_name)
    env.seed(seed)

    trajs = []
    trajs_frames = []
    for trial in range(trials):
        valid = False
        while not valid:
            obs = env.reset()
            # if obs[0] > -0.707:
            #     continue
            traj = []
            traj_frames = []
            for step in range(timesteps):
                x = np.concatenate((obs, np.array([u])))
                traj.append(x)
                traj_frames.append(env.render(mode='rgb_array'))
                obs, _, _, _ = env.step([u]) # action
            traj_frames = np.stack(traj_frames)
            traj = np.stack(traj)
            if np.amax(traj[:, 2]) < env.max_speed - 0.001  and np.amin(traj[:, 2]) > -env.max_speed + 0.001:
                valid = True
        trajs_frames.append(traj_frames)
        trajs.append(traj)
    trajs = np.stack(trajs, axis=1) # (trials, timesteps, dim)
    trajs_frames = np.stack(trajs_frames, axis=1)
    # turn to tray scale
    trajs_frames = color.rgb2gray(np.reshape(trajs_frames, (-1, *trajs_frames.shape[2:])))
    trajs_frames = np.reshape(trajs_frames, (timesteps, trials, *trajs_frames.shape[1:]))
    tspan = np.arange(timesteps) * 0.05
    env.close()

    return trajs_frames, trajs, tspan, gym_settings


def get_dataset(seed=0, samples=50, test_split=0.5, save_dir=None, 
                us=[0], name='pendulum-gym-image-dataset.pkl', **kwargs):
    data = {}

    assert save_dir is not None
    # path = '{}/pendulum-small-angle-image-dataset.pkl'.format(save_dir)
    path = os.path.join(save_dir, name)
    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        trajs_frames_force = []
        trajs_force = []
        for u in us:
            trajs_frames, trajs, tspan, _ = sample_gym(seed=seed, trials=samples, u=u, **kwargs)
            trajs_frames_force.append(trajs_frames)
            trajs_force.append(trajs)
        # make a train/test split
        split_ix = int(samples * test_split)
        tmp = np.stack(trajs_frames_force, axis=0) # (n_u, n_ts, n_trial, 50, 50)
        data['x'], data['test_x'] = tmp[:,:,:split_ix,:,:], tmp[:,:,split_ix:,:,:]
        tmp = np.stack(trajs_force, axis=0) # (n_u, n_ts, n_trial, 3)
        data['obs'], data['test_obs'] = tmp[:,:,:split_ix,:], tmp[:,:,split_ix:,:]

        data['t'] = tspan
        data['us'] = us

        to_pickle(data, path)
    return data

if __name__ == "__main__":
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    # repreducibility
    seed = 0
    np.manual_seed = seed
    us = [0.0, -1.0, 1.0, -2.0, 2.0]
    ts = 20 ; ss = 512
    data = get_dataset(seed=seed, timesteps=ts,
                save_dir=THIS_DIR, us=us, samples=ss, name='pendulum-gym-image-dataset.pkl')

    train_data = {}
    train_data['x'] = data['x']
    train_data['obs'] = data['obs']
    train_data['t'] = data['t']
    train_data['us'] = data['us']
    to_pickle(train_data, os.path.join(THIS_DIR, 'pendulum-gym-image-dataset-train.pkl'))
    test_data = {}
    test_data['x'] = data['test_x']
    test_data['obs'] = data['test_obs']
    test_data['t'] = data['t']
    test_data['us'] = data['us']
    to_pickle(test_data, os.path.join(THIS_DIR, 'pendulum-gym-image-dataset-test.pkl'))