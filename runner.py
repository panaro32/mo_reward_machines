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

def plot_pf(file, dir='results', method='all', eval=False):
    if method == 'all':
        modes = ['pvi', 'pql', 'crm']
    else:
        modes = [method]

    if eval:
        evals = [False, True]
    else:
        evals = [False]

    x_max, y_max = None, None
    for mode in modes:
        for eval in evals:
            if eval and mode == 'pvi':
                continue

            if eval:
                path = Path(dir, f'{file}_{mode}_eval.csv')
                df = pd.read_csv(path)
                x = 'objective_1'
                y = 'objective_2'
                candidates = df.to_numpy()
            else:
                path = Path(dir, f'{file}_{mode}.csv')
                df = pd.read_csv(path)
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

            if (x_max, y_max) == (None, None):
                x_max, y_max = int(np.max(df[x])), int(np.max(df[y]))

            # Customize labels and title
            plt.title(f'{mode.upper()} {'Evaluation' if eval else 'Pareto'} Front', fontsize=20)
            plt.xlabel('Objective 1', fontsize=16)
            plt.ylabel('Objective 2', fontsize=16)
            plt.xlim(-0.05, x_max + 1.05)
            plt.ylim(-0.05, y_max + 1.05)

            # Save and show the result as a pdf
            if eval:
                path = Path(dir, f'{file}_{mode}_eval.pdf')
            else:
                path = Path(dir, f'{file}_{mode}.pdf')
            plt.savefig(path, bbox_inches='tight')
            plt.close()

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
    path = Path('results', f'{rm_files}_{mode}.csv')
    df.to_csv(path, index=False)

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


    plot_pf(['abb_once.rm',  'baa_once.rm'],  method='all', eval=True)
    plot_pf(['abb_once2.rm', 'baa_once2.rm'], method='all', eval=True)
    plot_pf(['abb_once.rm',  'baa_cycle.rm'], method='all', eval=True)
    plot_pf(['abb_once2.rm', 'baa_cycle.rm'], method='all', eval=True)
    plot_pf(['abb_cycle.rm', 'baa_cycle.rm'], method='all', eval=True)
