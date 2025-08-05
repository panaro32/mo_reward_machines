import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):
    """2D grid environment with discrete state and action spaces."""

    def __init__(self, size=5):
        self.size = size
        self.agent_pos = None
        self.buttonA_pos = None
        self.buttonB_pos = None
        self.observation_space = gym.spaces.MultiDiscrete([self.size]*2)
        self.action_space = gym.spaces.Discrete(4)
        self.movement = {
            0: np.array([ 1,  0]),
            1: np.array([ 0,  1]),
            2: np.array([-1,  0]),
            3: np.array([ 0, -1]),
        }
        self.model()

    def observe(self):
        return self.agent_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        self.buttonA_pos = np.array([1, self.size-2])
        self.buttonB_pos = np.array([self.size-2, 1])
        observation = self.observe()
        info = {}
        return observation, info

    def step(self, action):
        delta_pos = self.movement[action]
        self.agent_pos = np.clip(self.agent_pos + delta_pos, 0, self.size-1)
        observation = self.observe()
        reward = 0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def labels(self):
        prop_values = {}
        prop_values['a'] = np.array_equal(self.agent_pos, self.buttonA_pos)
        prop_values['b'] = np.array_equal(self.agent_pos, self.buttonB_pos)
        return prop_values

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

    def model(self):
        self.states = [(row, col) for row in range(self.size) for col in range(self.size)]
        self.actions = [act for act in range(4)]
        self.transitions = {}
        for state in self.states:
            for action in self.actions:
                next_state = np.clip(np.array(state) + self.movement[action], 0, self.size-1)
                self.transitions[(state, action)] = tuple(next_state)
        self.events = {state: {'a': False, 'b': False} for state in self.states}
        self.events[(1, self.size-2)]['a'] = True
        self.events[(self.size-2, 1)]['b'] = True
