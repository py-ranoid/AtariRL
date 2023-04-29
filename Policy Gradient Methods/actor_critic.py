import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, actor_dim=2, critic_dim=1, learning_rate=0.00025, gamma=0.985):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_actor = nn.Linear(hidden_dim, actor_dim)
        self.fc_critic = nn.Linear(hidden_dim, critic_dim)

        self.gamma = gamma
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.data = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

    def compute_actor(self, x, dim=0):
        x = self.forward(x)
        x = self.fc_actor(x)
        prob = F.softmax(x, dim=dim)
        return prob

    def compute_critic(self, x):
        x = self.forward(x)
        value = self.fc_critic(x)
        return value

    def process_memory(self):
        state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []

        for transition in self.data:
            state, action, reward, next_state, done = transition

            state_list.append(state)
            action_list.append([action])
            reward_list.append([reward / 100.0])
            next_state_list.append(next_state)
            done_mask = 0.0 if done else 1.0
            done_list.append([done_mask])

        state_array = np.array(state_list, dtype=np.float32)
        action_array = np.array(action_list)
        reward_array = np.array(reward_list, dtype=np.float32)
        next_state_array = np.array(next_state_list, dtype=np.float32)
        done_array = np.array(done_list, dtype=np.float32)

        state_batch = torch.from_numpy(state_array)
        action_batch = torch.from_numpy(action_array)
        reward_batch = torch.from_numpy(reward_array)
        next_state_batch = torch.from_numpy(next_state_array)
        done_batch = torch.from_numpy(done_array)

        self.data = []
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def process_memory_vtrace(self):
        s_lst, a_lst, r_lst, s_prime_lst, mu_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, mu_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            mu_a_lst.append([mu_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s_array = np.array(s_lst, dtype=np.float32)
        a_array = np.array(a_lst)
        r_array = np.array(r_lst, dtype=np.float32)
        s_prime_array = np.array(s_prime_lst, dtype=np.float32)
        done_array = np.array(done_lst, dtype=np.float32)
        mu_a_array = np.array(mu_a_lst)

        s = torch.from_numpy(s_array)
        a = torch.from_numpy(a_array)
        r = torch.from_numpy(r_array)
        s_prime = torch.from_numpy(s_prime_array)
        done_mask = torch.from_numpy(done_array)
        mu_a = torch.from_numpy(mu_a_array)
        self.data = []
        return s, a, r, s_prime, done_mask, mu_a

    def start_training(self):
        states, actions, rewards, next_states, dones = self.process_memory()
        target_values = rewards + self.gamma * self.compute_critic(next_states) * dones
        deltas = target_values - self.compute_critic(states)

        policy = self.compute_actor(states, dim=1)
        policy_action = policy.gather(1, actions)
        loss = -torch.log(policy_action) * deltas.detach() + F.smooth_l1_loss(self.compute_critic(states),
                                                                              target_values.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
