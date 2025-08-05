from collections import defaultdict
import numpy as np


def value_iteration_rm(rm, gamma=0.99, error=1e-6):
    value = {rm_state: 0 for rm_state in rm.nonterminal_states}
    max_error = np.inf
    while max_error > error:
        max_error = 0
        for curr_rm_state in rm.nonterminal_states:
            next_values = []
            for next_rm_state, reward_function in rm.state_transitions[curr_rm_state].values():
                reward = reward_function(defaultdict(int))
                next_value = value[next_rm_state] if next_rm_state != rm.terminal_state else 0
                next_values.append(reward + gamma * next_value)
            best_value = max(next_values)
            max_error = max(max_error, abs(best_value - value[curr_rm_state]))
            value[curr_rm_state] = best_value
    return value


def value_iteration_rm_env(rm_env, gamma=0.99, error=1e-6):
    env, rm = rm_env.env, rm_env.rm
    value = {(env_state, rm_state): 0 for env_state in env.states for rm_state in rm.nonterminal_states}
    max_error = np.inf
    while max_error > error:
        max_error = 0
        for curr_env_state in env.states:
            for curr_rm_state in rm.nonterminal_states:
                next_values = []
                for action in env.actions:
                    next_env_state = env.transitions[(curr_env_state, action)]
                    prop_values = env.events[next_env_state]
                    next_rm_state, reward, done = rm.step(curr_rm_state, prop_values)
                    next_value = value[(next_env_state, next_rm_state)] if not done else 0
                    next_values.append(reward + gamma * next_value)
                best_value = max(next_values)
                max_error = max(max_error, abs(best_value - value[(curr_env_state, curr_rm_state)]))
                value[(curr_env_state, curr_rm_state)] = best_value
    return value
