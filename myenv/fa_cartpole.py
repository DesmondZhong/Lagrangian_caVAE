# Modified from OpenAI gym CartPole-v1 task
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp


class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 0.2
        self.masspole = 0.5
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.7 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.05  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 100
        self.MAX_VEL = 100 * np.pi

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            1.0,
            1.0,
            np.finfo(np.float32).max,
            self.MAX_VEL,])
        
        high_a = np.array([15.0, 15.0])
        low_a = -high_a
        self.action_space = spaces.Box(low=low_a, high=high_a, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def dynamics(self, t, y, u):
        f = np.zeros_like(y)
        costheta = math.cos(y[2])
        sintheta = math.sin(y[2])
        temp = (u[0] + self.polemass_length * y[3] * y[3] * sintheta) / self.total_mass
        thetaacc = (u[1] + self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        f[0] = y[1]
        f[1] = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        f[2] = y[3]
        f[3] = thetaacc
        return f


    def step(self, u):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        # costheta = math.cos(theta)
        # sintheta = math.sin(theta)
        # temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        # xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        # if self.kinematics_integrator == 'euler':
        #     x  = x + self.tau * x_dot
        #     x_dot = x_dot + self.tau * xacc
        #     theta = theta + self.tau * theta_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        # else: # semi-implicit euler
        #     x_dot = x_dot + self.tau * xacc
        #     x  = x + self.tau * x_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        #     theta = theta + self.tau * theta_dot
        ivp = solve_ivp(fun=lambda t, y:self.dynamics(t, y, u), t_span=[0, self.tau], y0=self.state)
        self.state = ivp.y[:, -1]
        # self.state = (x,x_dot,theta,theta_dot)
        # done =  x < -self.x_threshold \
        #         or x > self.x_threshold \
        #         or theta < -self.theta_threshold_radians \
        #         or theta > self.theta_threshold_radians
        done = False

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, np.cos(theta), np.sin(theta), x_dot, theta_dot])
    
    def reset(self):
        x = self.np_random.uniform(low=-2.2, high=2.2)
        x_dot = self.np_random.uniform(low=-0.1, high=0.1)
        theta = self.np_random.uniform(low=-3.14, high=3.14)
        theta_dot = self.np_random.uniform(low=-0.2, high=0.2)
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps_beyond_done = None
        return self._get_obs()

    def render(self, mode='human'):
        screen_width = 64
        screen_height = 64

        world_width = 2.4*2
        scale = screen_width/world_width
        carty = 32 # 
        polewidth = 3.0
        polelen = scale * (2 * self.length)
        cartwidth = 9.0
        cartheight = 6.0

        if self.viewer is None:
            from myenv import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset = 0 # cartheight/2.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(1, 0, 0)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(0, 1, 0)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            # self.axle = rendering.make_circle(polewidth/2)
            # self.axle.add_attr(self.poletrans)
            # self.axle.add_attr(self.carttrans)
            # self.axle.set_color(.5,.5,.8)
            # self.viewer.add_geom(self.axle)
            # self.track = rendering.Line((0,carty), (screen_width,carty))
            # self.track.set_color(0,0,0)
            # self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
