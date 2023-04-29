from actor_critic import *


class PPO(nn.Module):
    def __init__(self, learning_rate=0.0002, K_epoch=3, gamma=0.98, lmbda=0.95, eps_clip=0.1):
        super(PPO, self).__init__()
        self.actor_critic = ActorCritic()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.K_epoch = K_epoch
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip

    def compute_advantage(self, delta):
        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        return torch.tensor(advantage_lst, dtype=torch.float)

    def start_training(self):
        states, actions, rewards, next_states, done_masks, old_probs = self.actor_critic.process_memory_vtrace()

        for i in range(self.K_epoch):
            td_targets = rewards + self.gamma * self.actor_critic.compute_critic(next_states) * done_masks
            deltas = td_targets - self.actor_critic.compute_critic(states)
            deltas = deltas.detach().numpy()

            advantages = self.compute_advantage(deltas)

            new_probs = self.actor_critic.compute_actor(states, dim=1)
            new_probs = new_probs.gather(1, actions)
            ratios = torch.exp(torch.log(new_probs) - torch.log(old_probs))

            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2)
            critic_loss = F.smooth_l1_loss(self.actor_critic.compute_critic(states), td_targets.detach())
            total_loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            total_loss.mean().backward()
            self.optimizer.step()
