# Modified from OpenAI gym Pendulum-v0 task
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0):
        self.max_speed=100.
        self.max_torque=10.
        self.dt=.05
        self.g = g
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def dynamics(self, t, y, u):
        g = self.g
        m = 1.
        l = 1.

        f = np.zeros_like(y)
        f[0] = y[1]
        f[1] = (-3*g/(2*l) * np.sin(y[0] + np.pi) + 3./(m*l**2)*u)
        return f

    def step(self,u):
        th, thdot = self.state # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        ivp = solve_ivp(fun=lambda t, y:self.dynamics(t, y, u), t_span=[0, self.dt], y0=self.state)
        self.state = ivp.y[:, -1]

        # newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        # newth = th + newthdot*dt
        # newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        # self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 0.5])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from myenv import rendering
            self.viewer = rendering.Viewer(32,32)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(1, 1, 1)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            # axle = rendering.make_circle(.05)
            # axle.set_color(0,0,0)
            # self.viewer.add_geom(axle)
            # fname = path.join(path.dirname(__file__), "assets/empty.png")
            # self.img = rendering.Image(fname, 1., 1.)
            # self.imgtrans = rendering.Transform()
            # self.img.add_attr(self.imgtrans)

        # self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        # if self.last_u:
        #     self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)