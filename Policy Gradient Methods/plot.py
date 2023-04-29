import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

# Assuming the arrays are stored in a list called `rewards_arrays`
rewards_arrays = [np.load('./rewards/acer_reward.npy'),
                  np.load('rewards/acer_reward_512.npy'),
                  np.load('./rewards/acer_reward_1024.npy'),
                  np.load('./rewards/actor_critic_reward.npy'),
                  np.load('./rewards/actor_critic_reward_512.npy'),
                  np.load('./rewards/actor_critic_reward_1024.npy'),]

# Define colors and labels for the 6 arrays
colors = ['grey', 'lightcoral', 'peru', 'darkkhaki', 'crimson', 'orange']
labels = ['ACER', 'ACER with 512 hidden dimensions', 'ACER with 1024 hidden dimensions',
          'Actor-Critic', 'Actor-Critic with 512 hidden dimensions', 'Actor-Critic with 1024 hidden dimensions']

# Generate the plot
plt.figure(figsize=(10, 6))
smooth_term = 50

for i, rewards in enumerate(rewards_arrays):
    mean_rewards = []
    std_rewards = []
    episodes = list(range(0, len(rewards), smooth_term))
    for start_idx in episodes:
        end_idx = start_idx + smooth_term
        mean_rewards.append(np.mean(rewards[start_idx:end_idx]))
        std_rewards.append(np.std(rewards[start_idx:end_idx])/smooth_term * 2)

    mean_rewards = np.array(mean_rewards)
    std_rewards = np.array(std_rewards)

    plt.plot(episodes, mean_rewards, label=labels[i], color=colors[i], linewidth=2)
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
                     color=colors[i], alpha=0.15)

# Customize the plot
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Average Reward', fontsize=14)
plt.title('Rewards of policy gradient algorithms in the CartPole game', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 1000)

# Show the plot
plt.show()
