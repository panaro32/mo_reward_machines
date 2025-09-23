from pathlib import Path
from collections import defaultdict
from itertools import product
import sympy.logic.boolalg as smp
import numpy as np
import gymnasium as gym
import pygraphviz as pgv


class RewardMachine:
    """Represent a non-Markovian multi-objective reward function using a finite state machine."""

    def __init__(self, rm_data=()):
        self.nonterminal_states = set()
        self.initial_state = None
        self.terminal_state = -1
        self.prop_symbols = set()
        self.state_transitions = defaultdict(dict)
        if isinstance(rm_data, (str, Path)):
            self.num_objectives = 1
            self.name = Path(rm_data)
            self.load(rm_data)
        else:
            self.num_objectives = len(rm_data)
            path = Path(rm_data[0]).parent if rm_data else Path('rm_files')
            self.name = Path(path, f"morm({'+'.join(Path(file).stem for file in rm_data)}).rm")
            self.combine(rm_data)
        self.nonterminal_states = sorted(self.nonterminal_states)
        self.prop_symbols = sorted(map(str, self.prop_symbols))
        self.validate()

    def load(self, rm_file):
        with open(rm_file) as rm_file:
            self.initial_state, terminal_states = eval(rm_file.readline())
            self.nonterminal_states.add(self.initial_state)
            terminal_states = set(terminal_states) | {self.terminal_state}
            for curr_state, next_state, prop_formula, reward_function in map(eval, rm_file):
                if curr_state in terminal_states:
                    continue
                self.nonterminal_states.add(curr_state)
                if next_state in terminal_states:
                    next_state = self.terminal_state
                else:
                    self.nonterminal_states.add(next_state)
                prop_formula = smp.simplify_logic(prop_formula)
                self.prop_symbols.update(prop_formula.free_symbols)
                self.state_transitions[curr_state][prop_formula] = (next_state, reward_function)

    def combine(self, rm_files):
        rms = tuple(RewardMachine(rm_file) for rm_file in rm_files)
        self.initial_state = tuple(rm.initial_state for rm in rms)
        reachable_states = {self.initial_state}
        while reachable_states:
            curr_state = reachable_states.pop()
            self.nonterminal_states.add(curr_state)
            rms_transitions = tuple(rm.state_transitions[rm_curr_state] for rm, rm_curr_state in zip(rms, curr_state))
            for rms_prop_formulas in product(*(rm_transition.keys() for rm_transition in rms_transitions)):
                prop_formula = smp.simplify_logic(smp.And(*rms_prop_formulas))
                self.prop_symbols.update(prop_formula.free_symbols)
                if prop_formula == smp.false:
                    continue
                next_state = tuple(rm_tran[rm_form][0] for rm_tran, rm_form in zip(rms_transitions, rms_prop_formulas))
                if any(rm_next_state == rm.terminal_state for rm, rm_next_state in zip(rms, next_state)):
                    next_state = self.terminal_state
                elif next_state not in self.nonterminal_states:
                    reachable_states.add(next_state)
                rms_reward_functions = tuple(rm_tran[rm_form][1] for rm_tran, rm_form in zip(rms_transitions, rms_prop_formulas))
                reward_function = lambda r, _rfs=rms_reward_functions: [rm_reward_function(r) for rm_reward_function in _rfs]
                self.state_transitions[curr_state][prop_formula] = (next_state, reward_function)

    def validate(self):
        for curr_state in self.nonterminal_states:
            prop_formulas = self.state_transitions[curr_state].keys()
            assert all(prop_formula != smp.false for prop_formula in prop_formulas), f'unsatisfiable formula for state {curr_state}'
            assert smp.simplify_logic(smp.Exclusive(*prop_formulas)) == smp.true, f'not mutually exclusive formulas for state {curr_state}'
            assert smp.simplify_logic(smp.Or(*prop_formulas)) == smp.true, f'not collectively exhaustive formulas for state {curr_state}'

    def reset(self):
        return self.initial_state

    def step(self, curr_state, prop_values, env_info=defaultdict(int)):
        for prop_formula, (next_state, reward_function) in self.state_transitions[curr_state].items():
            if prop_formula.subs(prop_values) == smp.true:
                return next_state, reward_function(env_info), next_state == self.terminal_state

    def ilabels(self):
        text = input(f"Active Labels ({'/'.join(self.prop_symbols)}): ")
        value = lambda prop_symbol: prop_symbol in text
        prop_values = {prop_symbol: value(prop_symbol) for prop_symbol in self.prop_symbols}
        return prop_values

    def __repr__(self):
        lines = [f'Reward Machine: {self.name}', f'#Objectives: {self.num_objectives}, #States: {len(self.nonterminal_states)}, #Symbols: {len(self.prop_symbols)}']
        width = max(len(str(self.initial_state)), len(str(self.terminal_state)))
        for curr_state in self.nonterminal_states:
            for prop_formula, (next_state, reward_function) in self.state_transitions[curr_state].items():
                lines.append(f'{curr_state!s:{width}} -> {next_state!s:{width}} $ {reward_function(defaultdict(int))} if {prop_formula}')
        return '\n'.join(lines)

    def draw(self):
        graph = pgv.AGraph(strict=False, directed=True, rankdir='LR')
        graph.node_attr.update(shape='circle', width=0.5)
        graph.add_node('start', shape='none', width=0, label='')
        graph.add_edge('start', self.initial_state, label='start')
        for curr_state in self.nonterminal_states:
            for prop_formula, (next_state, reward_function) in self.state_transitions[curr_state].items():
                if next_state == self.terminal_state:
                    next_state = (next_state, curr_state, prop_formula)
                    graph.add_node(next_state, shape='point', width=0.1, label='')
                graph.add_edge(curr_state, next_state, label=f'{prop_formula}, {reward_function(defaultdict(int))}')
        graph.draw(self.name.with_suffix('.png'), prog='dot', args='-Gdpi=300')


class RewardMachineEnv(gym.Wrapper):
    """Augment a labeled environment with a multi-objective reward machine."""

    def __init__(self, env, rm, interactive=False):
        super().__init__(env)
        self.rm = rm
        self.labels = self.rm.ilabels if interactive else self.env.unwrapped.labels
        self.curr_env_state = None
        self.curr_rm_state = None
        env_obs_space = self.env.observation_space
        rm_obs_space = gym.spaces.MultiBinary(len(self.rm.nonterminal_states))
        self.dict_space = gym.spaces.Dict({'env': env_obs_space, 'rm': rm_obs_space})
        self.observation_space = gym.spaces.utils.flatten_space(self.dict_space)
        self.rm_obs = dict(zip(self.rm.nonterminal_states, np.identity(len(self.rm.nonterminal_states))))
        self.rm_obs[self.rm.terminal_state] = np.zeros(len(self.rm.nonterminal_states))

    def observe(self, env_state, rm_state, done=False):
        if done:
            rm_state = self.rm.terminal_state
        dict_obs = {'env': env_state, 'rm': self.rm_obs[rm_state]}
        return gym.spaces.utils.flatten(self.dict_space, dict_obs)

    def reset(self, seed=None, options=None):
        self.curr_env_state, info = self.env.reset(seed=seed)
        self.curr_rm_state = self.rm.reset()
        observation = self.observe(self.curr_env_state, self.curr_rm_state)
        print(f'RESET: S={self.curr_env_state}, U={self.curr_rm_state}, O={observation}')
        return observation, info

    def step(self, action):
        next_env_state, env_reward, env_done, truncated, info = self.env.step(action)
        prop_values = self.labels()
        info['crm'] = []
        for crm_state in self.rm.nonterminal_states:
            crm_obs = self.observe(self.curr_env_state, crm_state)
            next_crm_state, crm_reward, crm_done = self.rm.step(crm_state, prop_values, info)
            next_crm_obs = self.observe(next_env_state, next_crm_state, env_done or crm_done)
            info['crm'].append((crm_obs, action, np.array(crm_reward), next_crm_obs, env_done or crm_done))
        self.curr_env_state = next_env_state
        self.curr_rm_state, rm_reward, rm_done = self.rm.step(self.curr_rm_state, prop_values, info)
        observation = self.observe(self.curr_env_state, self.curr_rm_state, env_done or rm_done)
        print(f'STEP: S={self.curr_env_state}, U={self.curr_rm_state}, O={observation}, R={rm_reward}, D={env_done or rm_done}')
        return observation, np.array(rm_reward), env_done or rm_done, truncated, info

    def iaction(self):
        actions = range(self.env.action_space.start, self.env.action_space.start + self.env.action_space.n)
        text = input(f"Agent Action ({'/'.join(map(str, actions))}): ")
        action = int(text)
        return action
