from pathlib import Path
import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
import seaborn as sns
import matplotlib.pyplot as plt

from reward_machines import RewardMachine, RewardMachineEnv
from labeled_envs import GridWorldEnv
from pvi import pvi, pvi_rm
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
import pandas as pd
from morl_baselines.common.pareto import get_non_pareto_dominated_inds


gym.register(
    id='GridWorld',
    entry_point=GridWorldEnv,
    max_episode_steps=200,
)

def plot_pf(file, method ='all', eval = False):
    if method == 'all':
        modes = ['pql', 'pvi', 'crm']
    else:
        modes = [method]
    for mode in modes:
        if eval:
            df = pd.read_csv(f'{file}_{mode}_eval.csv')
            x = 'objective_1'
            y = 'objective_2'
            candidates = df.to_numpy()
        else:
            df = pd.read_csv(f'{file}_{mode}.csv')
            x = 'Objective 1'
            y = 'Objective 2'
            candidates = df.iloc[:, 1:].to_numpy()
        nd_inds = get_non_pareto_dominated_inds(candidates)
        df = df.loc[nd_inds]
        if mode == 'pql':
            color = 'blue'
        elif mode == 'crm':
            color = 'red'
        else:
            color = 'green'

        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            s=100,
            alpha=0.8,
            color=color,
        )

        # Customize labels and title
        plt.title(f'{mode.upper()} pareto front', fontsize=18)
        plt.xlabel('Objective 1', fontsize=14)
        plt.ylabel('Objective 2', fontsize=14)
        plt.xlim(-0.05, int(np.max(df[x])) + 1.1)
        plt.ylim(-0.05, int(np.max(df[y])) + 1.1)

        # Save and show the result as a pdf
        if eval:
            filename = f'{file}_{mode}_eval.pdf'
        else:
            filename = f'{file}_{mode}.pdf'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

def plot_pareto(file, mode='pql'):
    # if rm_env.unwrapped.reward_dim == 2:
    data = pd.read_csv(f'{file}_{mode}.csv')
    obj1 = data['Objective 1']
    obj2 = data['Objective 2']
    #obj1, obj2 = zip(*pareto)
    plt.scatter(x=obj1, y=obj2)
    plt.title(f'{mode.upper()} pareto front')
    plt.xlabel('objective_1')
    plt.xlim(0, int(max(obj1)) + 1)
    plt.ylabel('objective_2')
    plt.ylim(0, int(max(obj2)) + 1)

    plt.savefig(f'{file}_{mode}.pdf', bbox_inches='tight')

def run_experiment(rm_files, rm_path='rm_files', env_id='GridWorld', gamma=0.99, mode='pql', seed=42, runs=30):
    def make_env(use_crm=False, log=False):
        env = gym.make(env_id)
        rm = RewardMachine([Path(rm_path, rm_file) for rm_file in rm_files])
        rm_env = RewardMachineEnv(env, rm, use_crm=use_crm)
        if log: rm_env = mo_gym.wrappers.MORecordEpisodeStatistics(rm_env, gamma=gamma)
        return rm_env

    if mode == 'draw':
        rm_env = make_env(use_crm=False, log=False)
        rm_env.rm.draw()
        return rm_env
    data = []
    columns = ["Run", "Objective 1", "Objective 2"]
    if mode in ['pvi', 'pvi_rm']:
        rm_env = make_env(use_crm=False, log=False)
        if mode == 'pvi':
            pareto = pvi(rm_env, gamma=gamma)
        else:
            pareto = pvi_rm(rm_env.rm, gamma=gamma)

        for el in pareto:
            # add el to data
            data.append([seed, el[0], el[1]])
    if mode in ['pql', 'crm']:
        for run in range(runs):
            train_env = make_env(use_crm=mode=='crm', log=True)
            test_env = make_env(use_crm=False, log=True)
            TIME = 50000
            agent = PQL(
                env=train_env,
                gamma=gamma,
                ref_point=np.ones(train_env.unwrapped.reward_dim)*(-0.5),
                initial_epsilon=1.0,
                epsilon_decay_steps=TIME,
                final_epsilon=0.1,
                seed=(seed+run),
                project_name=f'{rm_files}',
                experiment_name=f'{mode}',
                log=True,
            )
            pareto = agent.train(
                total_timesteps=TIME,
                eval_env=test_env,
                log_every=1000,
            )
            agent.close_wandb()
            # Save pareto set to csv

            for el in pareto:
                # add el to data
                data.append([seed+run, el[0], el[1]])

    # Keep in data only the entries that are not pareto dominated?
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f'{rm_files}_{mode}.csv', index=False)

    return pareto


if __name__ == '__main__':

    #print(run_experiment(['abb_once.rm',  'baa_once.rm'],  mode='pvi'))
    #print(run_experiment(['abb_once.rm',  'baa_once.rm'],  mode='pql'))
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


    plot_pf(['abb_once.rm', 'baa_once.rm'], 'all')
    plot_pf(['abb_once2.rm', 'baa_once2.rm'], 'all')
    plot_pf(['abb_once.rm',  'baa_cycle.rm'], 'all')
    plot_pf(['abb_once2.rm', 'baa_cycle.rm'], 'all')
    plot_pf(['abb_cycle.rm', 'baa_cycle.rm'], 'all')

    #plot_pf(['abb_once.rm', 'baa_cycle.rm'], 'pql', True)
    #plot_pf(['abb_once.rm', 'baa_cycle.rm'], 'crm', True)
