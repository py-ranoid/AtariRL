from matplotlib import pyplot as plt
import random
import collections
import gymnasium as gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyperparameters import *


def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.ylim(0, 1200)
    plt.show()


def compute_discounted_rewards(rewards, gamma):
    """
    Compute discounted rewards.
    :param rewards: List of rewards
    :param gamma: Discount factor
    :return: List of discounted rewards
    """
    discounted_rewards = []
    R = 0
    for reward in rewards[::-1]:
        R = reward + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards


def _process_mini_batch(mini_batch):
    state_list, action_list, reward_list, prob_list, done_list, is_first_list = [], [], [], [], [], []

    for sequence in mini_batch:
        is_first = True
        for transition in sequence:
            state, action, reward, prob, done = transition

            state_list.append(state)
            action_list.append([action])
            reward_list.append(reward)
            prob_list.append(prob)
            done_mask = 0.0 if done else 1.0
            done_list.append(done_mask)
            is_first_list.append(is_first)
            is_first = False

    state_array = np.array(state_list, dtype=np.float32)
    action_array = np.array(action_list)
    reward_array = np.array(reward_list, dtype=np.float32)
    prob_array = np.array(prob_list, dtype=np.float32)
    done_array = np.array(done_list, dtype=np.float32)

    state_tensor = torch.from_numpy(state_array)
    action_tensor = torch.from_numpy(action_array)
    reward_tensor = torch.from_numpy(reward_array)
    prob_tensor = torch.from_numpy(prob_array)
    done_tensor = torch.from_numpy(done_array)

    return state_tensor, action_tensor, reward_tensor, prob_tensor, done_tensor, is_first_list


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_num)

    def put(self, sequence_data):
        self.buffer.append(sequence_data)

    def sample(self, on_policy=False):
        mini_batch = self._get_mini_batch(on_policy)
        state_tensor, action_tensor, reward_tensor, prob_tensor, done_tensor, is_first_list = _process_mini_batch(
            mini_batch)
        return state_tensor, action_tensor, reward_tensor, prob_tensor, done_tensor, is_first_list

    def size(self):
        return len(self.buffer)

    def _get_mini_batch(self, on_policy):
        if on_policy:
            return [self.buffer[-1]]
        else:
            return random.sample(self.buffer, batch)


def compute_q_return(state_values, done_masks, rewards, rho_clamped, q_values_action, is_first_in_sequence):
    q_return = state_values[-1] * done_masks[-1]
    q_return_list = []
    for i in reversed(range(len(rewards))):
        q_return = rewards[i] + gamma_rate * q_return
        q_return_list.append(q_return.item())
        q_return = rho_clamped[i] * (q_return - q_values_action[i]) + state_values[i]

        if is_first_in_sequence[i] and i != 0:
            q_return = state_values[i - 1] * done_masks[i - 1]

    q_return_list.reverse()
    q_return = torch.tensor(q_return_list, dtype=torch.float).unsqueeze(1)
    return q_return


def compute_loss(q_return, q_values_action, state_values, policy, policy_action, rho, rho_clamped, c, q_values):
    correction_coefficient = (1 - c / rho).clamp(min=0)
    loss1 = -rho_clamped * torch.log(policy_action) * (q_return - state_values)
    loss2 = -correction_coefficient * policy.detach() * torch.log(policy) * (q_values.detach() - state_values)
    loss = loss1 + loss2.sum(1) + F.smooth_l1_loss(q_values_action, q_return)
    return loss


def compute_v_trace_values_and_advantage(rewards, values, next_values, vs_minus_v_xs_lst):
    vs_minus_v_xs = torch.tensor(vs_minus_v_xs_lst, dtype=torch.float)
    v_trace_values = vs_minus_v_xs[:-1] + values.numpy()
    next_v_trace_values = vs_minus_v_xs[1:] + next_values.numpy()
    advantage = rewards + gamma_rate * next_v_trace_values - values.numpy()

    return v_trace_values, advantage


def compute_vs_minus_v_xs_lst(delta, clipped_cs):
    vs_minus_v_xs_lst = []
    vs_minus_v_xs = 0.0
    vs_minus_v_xs_lst.append([vs_minus_v_xs])

    for i in range(len(delta) - 1, -1, -1):
        vs_minus_v_xs = gamma_rate * clipped_cs[i][0] * vs_minus_v_xs + delta[i][0]
        vs_minus_v_xs_lst.append([vs_minus_v_xs])
    vs_minus_v_xs_lst.reverse()

    return vs_minus_v_xs_lst


def compute_td_target_and_delta(rewards, next_values, values, done_masks, clipped_rhos):
    td_target = rewards + gamma_rate * next_values * done_masks
    delta = clipped_rhos * (td_target - values).numpy()
    return td_target, delta


def compute_ratio(policy_probs_action, behavior_policy_actions):
    ratio = torch.exp(torch.log(policy_probs_action) - torch.log(behavior_policy_actions))
    return ratio


