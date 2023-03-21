import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gym
import numpy as np
from gridworld import TwoDGridWorld
from agents import rand, montecarlo
import math
import random
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Gym basics : https://github.com/bentrevett/pytorch-rl/blob/master/0%20-%20Introduction%20to%20Gym.ipynb
    
def run_experiment(env, agent, n_episodes = 30):
    for episode in range(n_episodes):        
        episode_reward = 0
        done = False
        state = env.reset()
        agent.states = [state]
        agent.rewards = [0]
        while not done:            
            action = agent.act(env) if episode < n_episodes - 1 else agent.act(env, greedy=True)
            state, reward, done, truncated, info  = env.step(action)
            agent.log(state[0], action, reward)
            episode_reward += reward
        print(f'episode: {episode+1}, reward: {episode_reward}, num steps: {len(agent.states)}')
        if len(agent.states) < 20:
            print("Path:",agent.states)
        agent.update()
        print (np.array([agent.Q_s[i] for i in range(len(agent.Q_s))]).reshape(env.size,env.size).round(2))
        print('---------------------------------')

if __name__ == "__main__":
    # env = gym.make("CartPole-v0")
    env = TwoDGridWorld(size=6)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = montecarlo.MonteCarloAgent(action_space=range(4), state_space=range(env.size*env.size), epsilon=0.7)
    run_experiment(env, agent, 750)