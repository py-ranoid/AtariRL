import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from actor_critic import *
from argparse import ArgumentParser
from hyperparameters import *
from pg import *
from acer import *
from vtrace import *
import numpy as np
from PPO import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--policy", required=True, type=str, help="The policy gradient algorithm to use")
    args = parser.parse_args()
    env = gym.make('CartPole-v1')
    score = 0.0

    if args.policy == "actor_critic":
        model = ActorCritic()
    elif args.policy == "policy_gradient":
        model = Policy(4, 128, 2)
    elif args.policy == 'acer':
        model = Acer()
    elif args.policy == 'vtrace':
        model = Vtrace()
    elif args.policy == 'ppo':
        model = PPO()

    episode_reward = []
    for i in range(training_episode):
        done = False
        s, _ = env.reset()
        while not done:
            acer_memory = []
            for t in range(num):
                if args.policy == "policy_gradient":
                    prob = model(torch.from_numpy(s).float())
                elif args.policy == "actor_critic":
                    prob = model.compute_actor(torch.from_numpy(s).float())
                elif args.policy == "acer" or args.policy == "vtrace" or args.policy == "ppo":
                    prob = model.actor_critic.compute_actor(torch.from_numpy(s).float())

                a = Categorical(prob).sample().item()
                s_prime, r, done, info, _ = env.step(a)
                if args.policy == "policy_gradient":
                    model.data.append((r, prob[a]))
                elif args.policy == "actor_critic":
                    model.data.append((s, a, r, s_prime, done))
                elif args.policy == "acer":
                    acer_memory.append((s, a, r / 100.0, prob.detach().numpy(), done))
                elif args.policy == "vtrace" or args.policy == "ppo":
                    model.actor_critic.data.append((s, a, r / 100.0, s_prime, prob[a].item(), done))

                s = s_prime
                score += r

                if done:
                    break
            if args.policy == 'acer':
                model.memory.put(acer_memory)
                if model.memory.size() > 500:
                    model.start_training(on_policy=True)
                    model.start_training()
            else:
                model.start_training()

        if i % save_period == 0:
            average_reward = score / save_period
            print("Result: {0:.2f} in {1} episodes".format(average_reward, i))
            episode_reward.append(average_reward)
            score = 0.0
    env.close()
    # np.save(args.policy + '_reward', episode_reward)
