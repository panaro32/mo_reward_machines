from pathlib import Path
import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
import matplotlib.pyplot as plt

from reward_machines import RewardMachine, RewardMachineEnv
from labeled_envs import GridWorldEnv
from pvi import pvi, pvi_rm
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
import pandas as pd


gym.register(
    id='GridWorld',
    entry_point=GridWorldEnv,
    max_episode_steps=200,
)


def run_experiment(rm_files, rm_path='rm_files', env_id='GridWorld', gamma=0.99, mode='pql', seed=42, runs=5):
    def make_env(use_crm=False, log=False):
        env = gym.make(env_id)
        rm = RewardMachine([Path(rm_path, rm_file) for rm_file in rm_files])
        rm_env = RewardMachineEnv(env, rm, use_crm=use_crm)
        if log: rm_env = mo_gym.wrappers.MORecordEpisodeStatistics(rm_env, gamma=gamma)
        return rm_env
    def plot_pareto(pareto):
        #if rm_env.unwrapped.reward_dim == 2:
        obj1, obj2 = zip(*pareto)
        plt.scatter(x=obj1, y=obj2)
        plt.title(f'{mode.upper()} pareto front')
        plt.xlabel('objective_1')
        plt.xlim(0, int(max(obj1))+1)
        plt.ylabel('objective_2')
        plt.ylim(0, int(max(obj2))+1)
        plt.show()
    if mode == 'draw':
        rm_env = make_env(use_crm=False, log=False)
        rm_env.rm.draw()
        return rm_env
    if mode in ['pvi', 'pvi_rm']:
        rm_env = make_env(use_crm=False, log=False)
        if mode == 'pvi':
            pareto = pvi(rm_env, gamma=gamma)
        else:
            pareto = pvi_rm(rm_env.rm, gamma=gamma)
        columns = ["Run", "Objective 1", "Objective 2"]
        data = []
        for el in pareto:
            # add el to data
            data.append([seed, el[0], el[1]])
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(f'{rm_files}_{mode}_{seed}.csv', index=False)
        #plot_pareto(pareto)
        return pareto
    if mode in ['pql', 'crm']:
        for run in range(runs):
            train_env = make_env(use_crm=mode=='crm', log=True)
            test_env = make_env(use_crm=False, log=True)
            TIME = 5000
            agent = PQL(
                env=train_env,
                gamma=gamma,
                ref_point=np.zeros(train_env.unwrapped.reward_dim),
                initial_epsilon=1.0,
                epsilon_decay_steps=TIME,
                final_epsilon=0.1,
                seed=seed+run,
                project_name=f'{rm_files}',
                experiment_name=f'{mode}',
                log=False,
            )
            pareto = agent.train(
                total_timesteps=TIME,
                eval_env=test_env,
                log_every=100,
            )
            # Save pareto set to csv
            columns = ["Run", "Objective 1", "Objective 2"]
            data = []
            for el in pareto:
                # add el to data
                data.append([seed+run, el[0], el[1]])
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(f'{rm_files}_{mode}_{seed}.csv', index=False)
        print(pareto)
        #agent.close_wandb()


        #plot_pareto(pareto)
        return pareto


if __name__ == '__main__':
    pass

    #print(run_experiment(['abb_once.rm',  'baa_once.rm'],  mode='pvi'))
    print(run_experiment(['abb_once.rm',  'baa_once.rm'],  mode='pql'))
    #print(run_experiment(['abb_once.rm',  'baa_once.rm'],  mode='crm'))

    #print(run_experiment(['abb_once2.rm', 'baa_once2.rm'], mode='pvi'))
    #print(run_experiment(['abb_once2.rm', 'baa_once2.rm'], mode='pql'))
    #print(run_experiment(['abb_once2.rm', 'baa_once2.rm'], mode='crm'))

    #print(run_experiment(['abb_once.rm',  'baa_cycle.rm'], mode='pvi'))
    #print(run_experiment(['abb_once.rm',  'baa_cycle.rm'], mode='pql'))
    #print(run_experiment(['abb_once.rm',  'baa_cycle.rm'], mode='crm'))

    #print(run_experiment(['abb_once2.rm', 'baa_cycle.rm'], mode='pvi'))
    #print(run_experiment(['abb_once2.rm', 'baa_cycle.rm'], mode='pql'))
    #print(run_experiment(['abb_once2.rm', 'baa_cycle.rm'], mode='crm'))

    #print(run_experiment(['abb_cycle.rm', 'baa_cycle.rm'], mode='pvi'))
    #print(run_experiment(['abb_cycle.rm', 'baa_cycle.rm'], mode='pql'))
    #print(run_experiment(['abb_cycle.rm', 'baa_cycle.rm'], mode='crm'))
