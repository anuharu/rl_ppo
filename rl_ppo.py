import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
import wandb
import gym
from collections import defaultdict
import pandas as pd
from difflib import SequenceMatcher

unlikely_patterns_df = pd.read_csv("unlikely_patterns.csv")
unlikely_patterns = unlikely_patterns_df['Pattern'].tolist()
commonsld_df = pd.read_csv('commonsld.csv')
real_sld_data = dict(zip(commonsld_df['sld'], commonsld_df['occurrence_count']))

alphabet = list(set("abcdefghijklmnopqrstuvwxyz")) + ["_", "<END>"]
max_length = 12

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end_of_word = False
        self.end_count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, count=1):
        node = self.root
        for char in word:
            node = node.children[char]
        node.is_end_of_word = True
        node.end_count += count

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False, 0
            node = node.children[char]
        return node.is_end_of_word, node.end_count if node.is_end_of_word else 0

    def get_occurrence_count(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.end_count if node.is_end_of_word else 0

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Initialize trie with data from commonsld.csv
trie = Trie()
for sld, count in real_sld_data.items():
    trie.insert(sld, count)

class SLDReward:
    def __init__(self, trie, unlikely_patterns, real_sld_data, max_length):
        self.trie = trie
        self.unlikely_patterns = unlikely_patterns
        self.real_sld_data = real_sld_data
        self.max_length = max_length

    def _calculate_reward(self, sld_str):
        reward = 10  # Baseline reward
        is_valid, end_count = self.trie.search(sld_str)
        valid_word_bonus = 300 if is_valid else 0
        occurrence_bonus = min(end_count * 10, 100)
        penalty = sum(-50 for pattern in self.unlikely_patterns if pattern in sld_str)
        length_penalty = -10 * max(0, len(sld_str) - self.trie.root.end_count)

        if len(sld_str) < 5:
            length_bonus = -20 * (5 - len(sld_str))
        elif len(sld_str) > 12:
            length_bonus = -10 * (len(sld_str) - 12)
        else:
            length_bonus = 50

        valid_prefix_bonus = 0
        if self.trie.starts_with(sld_str):
            valid_prefix_bonus = len(sld_str) * 2
            valid_prefix_bonus *= len(sld_str) / self.max_length

        reward += valid_word_bonus + valid_prefix_bonus + occurrence_bonus + penalty + length_bonus
        reward = max(-200, reward)

        wandb.log({
            "reward_valid_word": valid_word_bonus,
            "reward_occurrence": occurrence_bonus,
            "reward_valid_prefix": valid_prefix_bonus,
            "penalty_unlikely_patterns": penalty,
            "length_bonus": length_bonus,
            "total_reward": reward,
        })

        return reward, end_count

# Custom environment
class SLDEnvironment(gym.Env):
    def __init__(self, alphabet, max_length, trie, unlikely_patterns, real_sld_data):
        super(SLDEnvironment, self).__init__()
        self.alphabet = alphabet
        self.max_length = max_length
        self.current_sld = []
        self.reward_calculator = SLDReward(trie, unlikely_patterns, real_sld_data, max_length)
        self.action_space = gym.spaces.Discrete(len(alphabet))
        self.observation_space = gym.spaces.MultiDiscrete([len(alphabet)] * max_length)

    def reset(self):
        self.current_sld = []
        return [0] * self.max_length

    def step(self, action):
        if action == len(self.alphabet) - 1: 
            done = True
            reward, max_similarity = self.reward_calculator._calculate_reward(''.join(self.current_sld))
            state = [self.alphabet.index(c) for c in self.current_sld] + [0] * (self.max_length - len(self.current_sld))
            return np.array(state[:self.max_length]), reward, done, {"max_similarity": max_similarity}
        
        char = self.alphabet[action]
        self.current_sld.append(char)
        done = len(self.current_sld) >= self.max_length
        reward, max_similarity = self.reward_calculator._calculate_reward(''.join(self.current_sld))
        state = [self.alphabet.index(c) for c in self.current_sld] + [0] * (self.max_length - len(self.current_sld))
        return np.array(state[:self.max_length]), reward, done, {"max_similarity": max_similarity}

    def render(self):
        return ''.join(self.current_sld)

class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, vocab_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        embeddings = self.embedding(x)
        lstm_out, _ = self.lstm(embeddings)
        return lstm_out

    def get_action_and_value(self, x, action=None):
        x = x.to(next(self.parameters()).device)
        lstm_out = self.forward(x)
        logits = self.actor(lstm_out[:, -1, :])
        value = self.critic(lstm_out[:, -1, :])
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value, probs

    def get_value(self, x):
        x = x.to(next(self.parameters()).device)
        lstm_out = self.forward(x)
        return self.critic(lstm_out[:, -1, :])

# PPO Agent
class PPOAgent:
    def __init__(self, policy_net, optimizer, gamma=0.99, gae_lambda=0.95, clip_coef=0.2,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        # Initialize W&B
        wandb.init(project="SLD_PPO", name="SLD_generation_PPO")

    def compute_returns_and_advantages(self, rewards, values, dones, next_value):
        advantages = torch.zeros_like(rewards).to(values.device)
        returns = torch.zeros_like(rewards).to(values.device)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
        returns = advantages + values
        return returns, advantages

    def train(self, env, num_episodes=300, num_steps=max_length, update_epochs=4, minibatch_size=32):
        device = next(self.policy_net.parameters()).device
        obs_shape = env.observation_space.shape

        for episode in range(num_episodes):
            # Storage
            observations = torch.zeros((num_steps, max_length), dtype=torch.long).to(device)
            actions = torch.zeros((num_steps,), dtype=torch.long).to(device)
            log_probs = torch.zeros((num_steps,)).to(device)
            rewards = torch.zeros((num_steps,)).to(device)
            dones = torch.zeros((num_steps,)).to(device)
            values = torch.zeros((num_steps,)).to(device)

            obs = env.reset()
            obs = torch.tensor(obs, dtype=torch.long, device=device).unsqueeze(0)

            for t in range(num_steps):
                with torch.no_grad():
                    action, log_prob, _, value, _ = self.policy_net.get_action_and_value(obs)
                next_obs, reward, done, _ = env.step(action.item())
                next_obs = torch.tensor(next_obs, dtype=torch.long, device=device)
                observations[t] = obs.squeeze(0)
                actions[t] = action
                log_probs[t] = log_prob
                rewards[t] = reward
                dones[t] = done
                values[t] = value
                obs = next_obs.unsqueeze(0)

            with torch.no_grad():
                next_value = self.policy_net.get_value(obs)
            returns, advantages = self.compute_returns_and_advantages(rewards, values, dones, next_value)

            ev = explained_variance(values, returns)
            wandb.log({"explained_variance": ev})

            for _ in range(update_epochs):
                indices = torch.randperm(num_steps)
                for start in range(0, num_steps, minibatch_size):
                    end = start + minibatch_size
                    mb_indices = indices[start:end]
                    mb_obs = observations[mb_indices]
                    mb_actions = actions[mb_indices]
                    mb_log_probs = log_probs[mb_indices]
                    mb_returns = returns[mb_indices]
                    mb_advantages = advantages[mb_indices]

                    _, new_log_probs, entropy, new_values, _ = self.policy_net.get_action_and_value(mb_obs, mb_actions)
                    ratio = (new_log_probs - mb_log_probs).exp()
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    value_loss = (new_values - mb_returns).pow(2).mean()
                    entropy_loss = entropy.mean()

                    loss = pg_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            avg_reward = rewards.mean().item()
            wandb.log({
                "episode": episode,
                "avg_reward": avg_reward,
                "policy_loss": pg_loss.item(),
                "value_loss": value_loss.item(),
                "entropy_loss": entropy_loss.item(),
            })
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

def explained_variance(y_pred, y_true):
    var_y = torch.var(y_true)
    return 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)

# Generate names
def generate_names(policy_net, env, num_names=10, max_length=12, temperature=1.0):
    policy_net.eval()
    device = next(policy_net.parameters()).device
    generated_names = []
    trie_matches = 0

    for _ in range(num_names):
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.long).unsqueeze(0).to(device)
        name = []

        for _ in range(max_length):
            with torch.no_grad():
                action_probs = policy_net.get_action_and_value(obs)[4].probs
                action_dist = Categorical(logits=action_probs / temperature)
                action = action_dist.sample().item()

                if action == len(env.alphabet) - 1:
                    break

                name.append(env.alphabet[action])
                next_obs, _, done, _ = env.step(action)
                obs = torch.tensor(next_obs, dtype=torch.long).unsqueeze(0).to(device)

                if done:
                    break

        name_str = ''.join(name)
        generated_names.append(name_str)
        if env.reward_calculator.trie.search(name_str)[0]:
            trie_matches += 1

    trie_match_rate = trie_matches / num_names
    wandb.log({"trie_match_rate": trie_match_rate})

    return generated_names

env = SLDEnvironment(alphabet, max_length, trie, unlikely_patterns, real_sld_data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = PolicyNetwork(len(alphabet)).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
agent = PPOAgent(policy_net, optimizer)

agent.train(env, num_episodes=300, num_steps=max_length)

generated_names = generate_names(policy_net, env, num_names=100, max_length=max_length, temperature=1.0)
valid_names = [name for name in generated_names if env.reward_calculator.trie.search(name)[0]]
valid_name_rate = len(valid_names) / len(generated_names)
avg_length = sum(len(name) for name in generated_names) / len(generated_names)

wandb.log({
    "valid_name_rate": valid_name_rate,
    "average_name_length": avg_length,
})

print("Generated Names Evaluation:")
print(f"Valid Name Rate: {valid_name_rate:.2%}")
print(f"Average Name Length: {avg_length:.2f}")
print("Sample Names:", generated_names[:10])
