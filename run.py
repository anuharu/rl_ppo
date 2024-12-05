from env import Environment
from ppo import PPO
import string
import matplotlib.pyplot as plt

ALPHABET = ['#'] + list(string.ascii_lowercase) + ['_']
NUM_EPISODES = 3000

if __name__ == '__main__':
    env = Environment()
    pi = PPO(env, num_episodes=NUM_EPISODES)
    avg_backlog = pi.train()

    test_env = Environment()
    for i in range(20):
        out = pi.eval(test_env)
        generated = [ALPHABET[x] for x in out[:-1]]
        print(generated)

    plt.plot([i for i in range(NUM_EPISODES)], avg_backlog)
    plt.show()
