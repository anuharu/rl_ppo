import pdb
import string

import gymnasium as gym
from collections import defaultdict
import pandas as pd
import numpy as np


ALPHABET = ['#'] + list(string.ascii_lowercase) + ['_']

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end_of_word = False
        self.end_count = 0
        self.count = 0


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, count=1):
        node = self.root
        for char in word:
            node = node.children[char]
            node.count += 1
        node.is_end_of_word = True
        node.end_count += count

    def reward(self, prefix):
        node = self.root
        depth = 0
        value = 0
        for char in prefix[:prefix[-1]]:
            depth += 1
            if char == 0:
                if node.end_count <= 0:
                    value -= 50
                    return False, value, 0
                break
            c = ALPHABET[char]
            if c not in node.children:
                return False, 0, 0
            node = node.children[c]
            value += np.power(0.5, prefix[-1] - depth) * node.count
        if node.end_count > 0:
            value += 100
        return True, value, depth


class Environment(gym.Env):

    def __init__(self, max_length=12):
        commonsld_df = pd.read_csv('commonsld.csv')
        real_sld_data = dict(zip(commonsld_df['sld'], commonsld_df['occurrence_count']))

        self.max_length = max_length
        self.observation_space = gym.spaces.Box(
            low = np.array([0] * (self.max_length + 1)),
            high = np.array([27] * self.max_length + [self.max_length]),
            shape=(self.max_length + 1,),
            dtype=np.int32)
        self.action_space = gym.spaces.Discrete(28)

        self.trie = Trie()
        for sld, count in real_sld_data.items():
            self.trie.insert(sld, count)

    def reset(self):
        self.state = [0] * (self.max_length + 1)
        return self.state, {}

    def reward_function(self, state=None, next_state=None):
        found, value, depth = self.trie.reward(state)
        return np.power(0.8, self.max_length - state[-1]) * value

    def step(self, action):
        prev_state = self.state
        self.state[self.state[-1]] = action.item()
        done = False
        self.state[-1] += 1
        if self.state[-1] == self.max_length:
            done = True
        reward = self.reward_function(prev_state)
        return self.state, reward, done, False, {}


