from itertools import product
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):
    """2D grid environment with discrete state and action spaces."""

    def __init__(self, size=5):
        self.size = size
        self.agent_pos = None
        self.buttonA_pos = None
        self.buttonB_pos = None
        self.last_state = None
        self.last_action = None
        self.observation_space = gym.spaces.MultiDiscrete([self.size]*2)
        self.states = list(product(*(range(start, start + n) for start, n in zip(self.observation_space.start, self.observation_space.nvec))))
        self.initial_state = tuple(self.observation_space.start)
        self.action_space = gym.spaces.Discrete(4)
        self.actions = list(range(self.action_space.start, self.action_space.start + self.action_space.n))
        self.movement = {
            0: np.array([ 1,  0]), # U
            1: np.array([ 0,  1]), # R
            2: np.array([-1,  0]), # D
            3: np.array([ 0, -1]), # L
        }
        self.reset()

    def observe(self):
        return self.agent_pos

    def transitions(self, state, action):
        pos = np.array(state)
        delta_pos = self.movement[action]
        next_pos = np.clip(pos + delta_pos, 0, self.size-1)
        next_state = tuple(next_pos)
        next_states = {next_state: 1}
        return next_states

    def labeling(self, state, action, next_state):
        prop_values = {}
        prop_values['a'] = np.array_equal(next_state, self.buttonA_pos)
        prop_values['b'] = np.array_equal(next_state, self.buttonB_pos)
        return prop_values

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        self.buttonA_pos = np.array([1, self.size-2])
        self.buttonB_pos = np.array([self.size-2, 1])
        observation = self.observe()
        info = {}
        return observation, info

    def step(self, action):
        self.last_state = self.agent_pos
        self.last_action = action
        next_states = self.transitions(self.agent_pos, action)
        next_state = list(next_states)[np.random.choice(len(next_states), p=list(next_states.values()))]
        self.agent_pos = np.array(next_state)
        observation = self.observe()
        reward = 0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def labels(self):
        return self.labeling(self.last_state, self.last_action, self.agent_pos)

    def render(self):
        lines = []
        for row in reversed(range(self.size)):
            line = []
            for col in range(self.size):
                cell = np.array([row, col])
                if np.array_equal(cell, self.agent_pos):
                    line.append('X')
                elif np.array_equal(cell, self.buttonA_pos):
                    line.append('A')
                elif np.array_equal(cell, self.buttonB_pos):
                    line.append('B')
                else:
                    line.append('.')
            lines.append(' '.join(line))
        print('\n'.join(lines))
