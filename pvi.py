from collections import defaultdict
from itertools import product
import numpy as np


def pvi(rm_env, gamma=0.99, epsilon=0.01, size=10, runs=1000):
    """Compute the set of non-dominated vectors per augmented reward machine environment state using Pareto Value Iteration."""
    env, rm = rm_env.env, rm_env.rm
    pareto = {state: {(0,) * rm.num_objectives} for state in product(env.states, rm.nonterminal_states)}
    for run in range(runs):
        new_pareto = {}
        for env_state, rm_state in product(env.states, rm.nonterminal_states):
            candidate_vectors = set()
            for action in env.actions:
                next_state_paretos = []
                for next_env_state in env.transitions(env_state, action):
                    prop_values = env.labeling(env_state, action, next_env_state)
                    next_rm_state, reward, done = rm.step(rm_state, prop_values)
                    next_state_pareto = pareto[(next_env_state, next_rm_state)] if not done else {(0,) * rm.num_objectives}
                    next_state_paretos.append(next_state_pareto)
                for next_state_vectors in product(*next_state_paretos):
                    total_reward = np.zeros(rm.num_objectives)
                    for (next_env_state, transition_prob), next_state_vector in zip(env.transitions(env_state, action).items(), next_state_vectors):
                        prop_values = env.labeling(env_state, action, next_env_state)
                        next_rm_state, reward, done = rm.step(rm_state, prop_values)
                        total_reward += transition_prob * (reward + gamma * np.array(next_state_vector))
                    candidate_vectors.add(tuple(total_reward))
            new_pareto[(env_state, rm_state)] = best(candidate_vectors, size)
        if converged(pareto, new_pareto, epsilon):
            break
        pareto = new_pareto
    return pareto


def pvi_rm(rm, gamma=0.99, epsilon=0.01, size=10, runs=1000):
    """Compute the set of non-dominated vectors per reward machine state using Pareto Value Iteration."""
    pareto = {rm_state: {(0,) * rm.num_objectives} for rm_state in rm.nonterminal_states}
    for run in range(runs):
        new_pareto = {}
        for rm_state in rm.nonterminal_states:
            candidate_vectors = set()
            for next_rm_state, reward_function in rm.state_transitions[rm_state].values():
                next_state_pareto = pareto[next_rm_state] if next_rm_state != rm.terminal_state else {(0,) * rm.num_objectives}
                reward = reward_function(defaultdict(int))
                for next_state_vector in next_state_pareto:
                    total_reward = reward + gamma * np.array(next_state_vector)
                    candidate_vectors.add(tuple(total_reward))
            new_pareto[rm_state] = best(candidate_vectors, size)
        if converged(pareto, new_pareto, epsilon):
            break
        pareto = new_pareto
    return pareto


def best(candidates, size=10):
    pareto = non_dominated(candidates)
    if size is None:
        return pareto
    extra = len(pareto) - size
    if extra <= 0:
        return pareto
    pareto = np.array(list(pareto))
    pruned_pareto = pareto[crowding(pareto).argsort()[extra:]]
    return {tuple(point) for point in pruned_pareto}


def non_dominated(candidates):
    candidates = np.array(list(candidates))
    candidates = candidates[np.flip(candidates.sum(axis=1).argsort())]
    size = 0
    while size < len(candidates):
        mask = np.array([True] * len(candidates))
        mask[size+1:] = np.any(candidates[size+1:] > candidates[size], axis=1)
        candidates, size = candidates[mask], size + 1
    return {tuple(candidate) for candidate in candidates}


def crowding(pareto):
    size, objectives = pareto.shape
    ordering = pareto.argsort(axis=0)
    distances = np.zeros(size)
    for obj in range(objectives):
        min_idx, max_idx = ordering[0, obj], ordering[-1, obj]
        norm_factor = pareto[max_idx, obj] - pareto[min_idx, obj]
        distances[min_idx] = distances[max_idx] = np.inf
        for rank in range(1, size-1):
            prev_idx, idx, next_idx = ordering[rank-1, obj], ordering[rank, obj], ordering[rank+1, obj]
            distance = pareto[next_idx, obj] - pareto[prev_idx, obj]
            distances[idx] += distance / norm_factor
    return distances


def converged(pareto, new_pareto, epsilon=0.01):
    for state in pareto:
        for new_vector in new_pareto[state]:
            min_epsilon = np.inf
            for vector in pareto[state]:
                diff = np.abs(np.array(new_vector) - np.array(vector))
                min_epsilon = min(min_epsilon, np.max(diff))
            if min_epsilon > epsilon:
                return False
    return True
