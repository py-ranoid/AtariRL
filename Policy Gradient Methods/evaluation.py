import numpy as np

if __name__ == '__main__':
    rewards_arrays = [
        np.load('./rewards/acer_reward.npy'),
        np.load('./rewards/actor_critic_reward.npy'),
        np.load('./rewards/policy_gradient_reward.npy'),
        np.load('./rewards/ppo_reward.npy'),
        np.load('./rewards/vtrace_reward.npy'),
    ]

    # Calculate the last 100 episodes' average rewards and standard deviations
    last_100_rewards_stats = []
    for rewards in rewards_arrays:
        last_100_rewards = rewards[-100:]
        avg_reward = np.mean(last_100_rewards)
        std_reward = np.std(last_100_rewards)
        last_100_rewards_stats.append((avg_reward, std_reward))

    # Print the results
    labels = ['ACER', 'Actor Critic', 'Policy Gradient', 'PPO', 'V-trace']
    for i, (avg_reward, std_reward) in enumerate(last_100_rewards_stats):
        print(f"{labels[i]}: Average Reward = {avg_reward:.2f}, Standard Deviation = {std_reward:.2f}")
