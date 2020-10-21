from gym.envs.registration import register

register(
    id='MyPendulum-v0',
    entry_point='myenv.pendulum:PendulumEnv',
)

register(
    id='My_FA_Acrobot-v0',
    entry_point='myenv.fa_acrobot:AcrobotEnv',
)

register(
    id='My_FA_CartPole-v0',
    entry_point='myenv.fa_cartpole:CartPoleEnv',
)