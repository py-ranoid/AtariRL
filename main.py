import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gym
import numpy as np
from gridworld import TwoDGridWorld
from agents import rand, montecarlo, dqn

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count

#Gym basics : https://github.com/bentrevett/pytorch-rl/blob/master/0%20-%20Introduction%20to%20Gym.ipynb

def run_dqn(env, agent):
    episode_durations = []
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 200

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        print("=============="*6)
        print("Starting episode:",i_episode)
        for t in count():
            print('new act')
            action = agent.act(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            if t % 1 ==0 :
                print("Time :",t)
                print(observation, reward, terminated, truncated)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            print("updating")
            agent.update()
            print("updated")

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            print('fetching policy dicts')
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            
            print('fetched policy dicts')
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*dqn.TAU + target_net_state_dict[key]*(1-dqn.TAU)
            print('loading policy dicts')
            agent.target_net.load_state_dict(target_net_state_dict)
            print('loaded policy dicts')
            
            if done:
                print("EP:",i_episode, t+1)
                episode_durations.append(t + 1)
                break
            print("end")
    # dqn.plot_durations(episode_durations)

def run_experiment(env, agent, n_episodes = 300):
    for episode in range(n_episodes):        
        episode_reward = 0
        done = False
        state = env.reset()
        agent.states = [state]
        agent.rewards = [0]
        while not done:            
            action = agent.act(state) if episode < n_episodes - 1 else agent.act(state, greedy=True)
            state, reward, done, truncated, info  = env.step(action)

            agent.log(state[0], action, reward)
            episode_reward += reward
        print(f'episode: {episode+1}, reward: {episode_reward}, num steps: {len(agent.states)}')
        if len(agent.states) < 20:
            print("Path:",agent.states)
        agent.update()
        fin = np.array([agent.Q_s[i] for i in range(len(agent.Q_s))]).reshape(env.size,env.size).round(2)
        # fin = fin - fin.min()
        # fin = fin/fin.max()
        # print (fin.tolist())
        print(fin)
        print('---------------------------------')
    plt.imshow(fin,cmap='gray')

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    # env = TwoDGridWorld(size=4)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # agent = montecarlo.MonteCarloAgent(action_space=range(4), state_space=range(env.size*env.size), epsilon=0.7)
    print(env.action_space)
    agent = dqn.DQN_Agent(action_space=env.action_space, state_space=env.reset()[0])
    # run_experiment(env, agent, 200)
    run_dqn(env, agent)