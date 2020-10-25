import os, sys
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
import numpy as np
from utils import to_pickle, from_pickle
import gym
import myenv
from skimage import color

def sample_gym(seed=0, timesteps=10, trials=50, side=28, min_angle=0., 
              verbose=False, u=[0.0, 0.0], env_name='My_FA_Acrobot-v0'):
    
    gym_settings = locals()
    if verbose:
        print("Making a dataset of Acrobot observations.")
    env: gym.wrappers.time_limit.TimeLimit = gym.make(env_name)
    env.seed(seed)

    trajs = []
    trajs_frames = []
    for trial in range(trials):
        valid = False
        while not valid:
            obs = env.reset()
            traj = []
            traj_frames = []
            for step in range(timesteps):
                x = np.array([obs[0], obs[2], obs[1], obs[3], obs[4], obs[5], u[0], u[1]])
                traj.append(x)
                traj_frames.append(env.render(mode='rgb_array'))
                obs, _, _, _ = env.step(u) # action
            traj = np.stack(traj)
            traj_frames = np.stack(traj_frames)
            if np.amax(traj[:, 4]) < env.MAX_VEL_1 - 0.001  and np.amin(traj[:, 4]) > -env.MAX_VEL_1 + 0.001:
                if np.amax(traj[:, 5]) < env.MAX_VEL_2 - 0.001  and np.amin(traj[:, 5]) > -env.MAX_VEL_2 + 0.001:
                    valid = True
        trajs.append(traj)
        trajs_frames.append(traj_frames)
    trajs = np.stack(trajs, axis=1) # (timesteps, trails, 2)
    trajs_frames = np.stack(trajs_frames, axis=1) # timesteps, trails, 32, 32, 3
    # turn to gray scale
    # trajs_frames = color.rgb2gray(np.reshape(trajs_frames, (-1, *trajs_frames.shape[2:])))
    # trajs_frames = np.reshape(trajs_frames, (timesteps, trials, *trajs_frames.shape[1:]))
    tspan = np.arange(timesteps) * 0.05
    env.close()

    return trajs_frames, trajs, tspan, gym_settings


def get_dataset(seed=0, samples=50, test_split=0.5, save_dir=None, 
                us=[0], name='acrobot-gym-image-dataset-rgb-0.pkl', **kwargs):
    data = {}

    assert save_dir is not None
    path = save_dir + '/' + name
    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        trajs_frames_force = []
        trajs_force = []
        for u in us:
            trajs_frames, trajs, tspan, _ = sample_gym(seed=seed, trials=samples, u=u, **kwargs)
            trajs_frames = (np.moveaxis(trajs_frames, -1, -3) / 255.0)
            trajs_frames_force.append(trajs_frames)
            trajs_force.append(trajs)
        # make a train/test split
        split_ix = int(samples * test_split)
        tmp = np.stack(trajs_frames_force, axis=0) # (n_u, n_ts, n_trial, 3, 64, 64)
        data['x'], data['test_x'] = tmp[:,:,:split_ix], tmp[:,:,split_ix:]
        tmp = np.stack(trajs_force, axis=0) # (n_u, n_ts, n_trial, 3)
        data['obs'], data['test_obs'] = tmp[:,:,:split_ix,:], tmp[:,:,split_ix:,:]

        data['t'] = tspan
        data['us'] = us

        to_pickle(data, path)
    return data


def arrange_data(x, t, num_points=2):
    '''Arrange data to feed into neural ODE in small chunks'''
    # x has shape n_u, ts, bs, 3, 32, 32
    assert num_points>=2 and num_points<=len(t)
    x_stack = []
    for i in range(num_points):
        if i < num_points-1:
            x_stack.append(x[:, i:-num_points+i+1])
        else:
            x_stack.append(x[:, i:])
    x_stack = np.stack(x_stack, axis=1) # n_u, n_p, ts+1-n_p, bs, 32, 32
    x_stack = np.reshape(x_stack, 
                (x.shape[0], num_points, -1, *x.shape[3:])) # n_u, n_p, (ts+1-n_p)*bs, 32, 32
    t_eval = t[0:num_points]
    return x_stack, t_eval

if __name__ == "__main__":
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    np.random.seed(0)
    # load data
    us = [[0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 2.0], [0.0, -2.0],
            [1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]
    # us = [[0.0, 0.0]]
    ts = 20 ; ss = 512
    data = get_dataset(seed=0, timesteps=ts,
                save_dir=THIS_DIR, us=us, samples=ss, test_split=0.50,
                name='acrobot-gym-image-dataset-rgb-u9.pkl')

    train_data = {}
    train_data['x'] = data['x']
    train_data['obs'] = data['obs']
    train_data['t'] = data['t']
    train_data['us'] = data['us']
    to_pickle(train_data, THIS_DIR + '/' + 'acrobot-gym-image-dataset-rgb-u9-train.pkl')
    test_data = {}
    test_data['x'] = data['test_x']
    test_data['obs'] = data['test_obs']
    test_data['t'] = data['t']
    test_data['us'] = data['us']
    to_pickle(test_data, THIS_DIR + '/' + 'acrobot-gym-image-dataset-rgb-u9-test.pkl')