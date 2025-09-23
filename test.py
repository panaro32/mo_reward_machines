from pathlib import Path
import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
#from morl_baselines.multi_policy.capql.capql import CAPQL
#from morl_baselines.multi_policy.envelope.envelope import Envelope
#import morl_baselines.multi_policy as morl # ???
from reward_machines import RewardMachine, RewardMachineEnv
from labeled_envs import GridWorldEnv

gym.register(
    id='GridWorld',
    #id='gymnasium_env/GridWorld-v0',
    entry_point=GridWorldEnv,
    #entry_point='mo_gymnasium.envs.my_env_dir.my_env_file:MyEnv',
    max_episode_steps=200,
)


#env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
#env = gym.wrappers.RecordVideo(env, 'videos/demo', episode_trigger=lambda e: True)
#make_gif <- MORLbaselines tutorial


if __name__ == '__main__':

    def make_env():

        #rm_path, rm_files = 'rm_files', ['abb_once.rm', 'baa_cycle.rm']
        rm_path, rm_files = 'rm_files', ['terminal_a.rm', 'optional_b.rm']
        rm = RewardMachine([Path(rm_path, rm_file) for rm_file in rm_files])
        #rm.draw()

        env = gym.make('GridWorld')
        #env = gym.make('GridWorld', render_mode='human')
        #env = gym.make('MountainCar-v0')
        #env.labels = lambda: {}

        rm_env = RewardMachineEnv(env, rm)
        #rm_env = RewardMachineEnv(env, rm, interactive=True)
        #print(rm_env.reset())
        #for _ in range(3):
            #print(rm_env.step(rm_env.iaction()))

        # TODO: move inside RewardMachineEnv
        rm_env.set_wrapper_attr('reward_space', gym.spaces.Box(0, 1, shape=(2,)))
        rm_env.set_wrapper_attr('reward_dim', rm_env.get_wrapper_attr('reward_space').shape[0])

        rm_env = mo_gym.wrappers.MORecordEpisodeStatistics(rm_env, gamma=0.99)

        return rm_env

    train_env = make_env()
    test_env = make_env()

    agent = PQL(
        env=train_env,
        gamma=0.99,
        ref_point=np.array([0, 0]),
    )
    #agent = CAPQL(
    #    env=train_env,
    #    gamma=0.99,
    #)

    #agent = Envelope(
    #    env=train_env,
    #    gamma=0.99,
    #    initial_epsilon=1.0,
    #    final_epsilon=0.01,
    #    epsilon_decay_steps=50000,
    #    seed=42
    #    #num_sample_w=20,
    #)

    agent.train(
        total_timesteps=100000,
        eval_env=test_env,
        ref_point=np.array([0, 0]),
        #verbose=True,
    )
