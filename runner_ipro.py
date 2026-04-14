from pathlib import Path
import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
import seaborn as sns
import matplotlib.pyplot as plt

from reward_machines import RewardMachine, RewardMachineEnv
from labeled_envs import GridWorldEnv
from pvi import pvi, pvi_rm
from morl_baselines.multi_policy.ipro.ipro_2d import IPRO2D
import pandas as pd
from morl_baselines.common.pareto import get_non_pareto_dominated_inds


gym.register(
    id='GridWorld',
    entry_point=GridWorldEnv,
    max_episode_steps=200,
)
"""
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
"""
def run_experiment_ipro(rm_files, rm_path='rm_files', env_id='GridWorld', gamma=0.99, mode='ipro', seed=42, runs=30):
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
    if mode in ['ipro', 'crm']:
        for run in range(runs):
            train_env = mo_gym.wrappers.vector.MOSyncVectorEnv([lambda: make_env(use_crm=mode=='crm', log=True)])
            test_env = make_env(use_crm=False, log=True)
            TIME = 500000
            agent = IPRO2D(
                env = train_env, #(gym.Env): The environment to solve.
                direction = "maximize", #(str): The direction of the objectives, either "maximize" or "minimize".
                offset = 1, #(float): The offset to apply to the extrema.
                tolerance = 1e-6, #(float): The tolerance for the algorithm.
                max_iterations = None, #(int, optional): The maximum number of iterations to run the algorithm.
                ###update_freq = 1, #(int): The frequency of updating hypervolume improvement heuristic for computing the next referent.
                reset_agent = True, #(bool): Whether to reset the agent after each iteration.
                aug = 0.1, #(float): The augmentation factor for the AASF.
                scale = 100, #(float): The scale factor for the AASF.
                iter_total_timesteps = TIME, #(int): The total number of timesteps for each iteration.
                learning_rate = 2.5e-4, #(float): The learning rate for the PPO algorithm.
                num_steps = 128, #(int): The number of rollout steps to take.
                anneal_lr = True, #(bool): Whether to anneal the learning rate.
                gamma = gamma, #(float): The discount factor for the PPO algorithm.
                gae_lambda = 0.95, #(float): The lambda parameter for Generalized Advantage Estimation.
                num_minibatches = 4, #(int): The number of minibatches to use for training.
                update_epochs = 4, #(int): The number of epochs to update the policy.
                norm_adv = True, #(bool): Whether to normalize the advantages.
                clip_coef = 0.2, #(float): The clipping coefficient for the PPO algorithm.
                clip_vloss = True, #(bool): Whether to clip the value loss.
                ent_coef = 0.01, #(float): The entropy coefficient for the PPO algorithm.
                vf_coef = 0.5, #(float): The value function coefficient for the PPO algorithm.
                max_grad_norm = 0.5, #(float): The maximum gradient norm for the PPO algorithm.
                target_kl = None, #(float, optional): The target KL divergence for the PPO algorithm.
                mc_k = 32, #(int): The number of Monte Carlo samples to use for the AASF.
                device = "auto", #(torch.device or str): The device to use for training.
                log = True, #(bool): Whether to log the training progress.
                experiment_name = f'{mode}', #(str, optional): The name of the experiment for logging.
                project_name = f'IPRO_{rm_files}', #(str): The name of the project for logging.
                wandb_entity = None, #(str, optional): The entity for Weights & Biases logging.
                wandb_mode = "online", #(str): The mode for Weights & Biases logging, either "online", "offline", or "disabled".
                seed = (seed+run), #(int): The random seed for reproducibility.
                rng = None, #(np.random.Generator, optional): A random number generator for reproducibility.
            )
            pareto = agent.train(
                eval_env = test_env,
                ref_point = np.ones(test_env.unwrapped.reward_dim)*(-0.5),
                deterministic = True,
                extrema = None,
                #extrema = (np.ones(test_env.unwrapped.reward_dim)*(0.0), np.ones(test_env.unwrapped.reward_dim)*(1.0)),
                callback = None,
            )
            agent.close_wandb()
            # Save pareto set to csv

            # keep only PF values, ignore learned policies
            pareto = [value for value, _policy in pareto]

            for el in pareto:
                # add el to data
                data.append([seed+run, el[0], el[1]])

    # Keep in data only the entries that are not pareto dominated?
    df = pd.DataFrame(data, columns=columns)
    path = Path('results_ipro', f'{rm_files}_{mode}.csv')
    df.to_csv(path, index=False)

    return pareto


if __name__ == '__main__':

    print(run_experiment_ipro(['abb_once.rm',  'baa_once.rm'],  mode='ipro', runs=1))
    print(run_experiment_ipro(['abb_once2.rm', 'baa_once2.rm'], mode='ipro', runs=1))
    print(run_experiment_ipro(['abb_once.rm',  'baa_cycle.rm'], mode='ipro', runs=1))
    print(run_experiment_ipro(['abb_once2.rm', 'baa_cycle.rm'], mode='ipro', runs=1))
    print(run_experiment_ipro(['abb_cycle.rm', 'baa_cycle.rm'], mode='ipro', runs=1))
