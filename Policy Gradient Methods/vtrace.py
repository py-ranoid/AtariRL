from actor_critic import *
from utils import *


class Vtrace(nn.Module):
    def __init__(self, learning_rate=0.00055, clip_rho_threshold=1.05, clip_c_threshold=1.03):
        super(Vtrace, self).__init__()
        self.data = []
        self.actor_critic = ActorCritic()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.clip_rho_threshold = torch.tensor(clip_rho_threshold, dtype=torch.float)
        self.clip_c_threshold = torch.tensor(clip_c_threshold, dtype=torch.float)

    def compute_policy_probs_action(self, states, actions):
        policy_probs = self.actor_critic.compute_actor(states, dim=1)
        policy_probs_action = policy_probs.gather(1, actions)
        return policy_probs_action

    def compute_clipped_rhos_cs(self, ratio):
        clipped_rhos = torch.min(self.clip_rho_threshold, ratio)
        clipped_cs = torch.min(self.clip_c_threshold, ratio).numpy()
        return clipped_rhos, clipped_cs

    def compute_vtrace_values(self, states, actions, rewards, next_states, done_masks, behavior_policy_actions):
        with torch.no_grad():
            policy_probs_action = self.compute_policy_probs_action(states, actions)
            values, next_values = self.actor_critic.compute_critic(states), self.actor_critic.compute_critic(
                next_states)
            ratio = compute_ratio(policy_probs_action, behavior_policy_actions)

            clipped_rhos, clipped_cs = self.compute_clipped_rhos_cs(ratio)
            td_target, delta = compute_td_target_and_delta(rewards, next_values, values, done_masks, clipped_rhos)

            vs_minus_v_xs_lst = compute_vs_minus_v_xs_lst(delta, clipped_cs)
            v_trace_values, advantage = compute_v_trace_values_and_advantage(rewards, values, next_values,
                                                                             vs_minus_v_xs_lst)

        return v_trace_values, advantage, clipped_rhos

    def start_training(self):
        states, actions, rewards, next_states, done_masks, behavior_policy_actions = self.actor_critic.process_memory_vtrace()
        v_trace_values, advantages, clipped_rhos = self.compute_vtrace_values(states, actions, rewards, next_states,
                                                                              done_masks, behavior_policy_actions)

        policy_probs = self.actor_critic.compute_actor(states, dim=1)
        policy_probs_action = policy_probs.gather(1, actions)

        value_loss = F.smooth_l1_loss(self.actor_critic.compute_critic(states), v_trace_values)
        policy_loss = -clipped_rhos * torch.log(policy_probs_action) * advantages
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
