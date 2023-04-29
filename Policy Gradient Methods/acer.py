from utils import *
from actor_critic import ActorCritic


class Acer(nn.Module):
    def __init__(self):
        super(Acer, self).__init__()

        self.actor_critic = ActorCritic(input_dim=4, hidden_dim=1024, actor_dim=2, critic_dim=2)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.memory = ReplayBuffer()

    def start_training(self, on_policy=False):
        states, actions, rewards, action_probs, done_masks, is_first_in_sequence = self.memory.sample(on_policy)

        q_values = self.actor_critic.compute_critic(states)
        q_values_action = q_values.gather(1, actions)
        policy = self.actor_critic.compute_actor(states, dim=1)
        policy_action = policy.gather(1, actions)
        state_values = (q_values * policy).sum(1).unsqueeze(1).detach()

        rho = policy.detach() / action_probs
        rho_action = rho.gather(1, actions)
        rho_clamped = rho_action.clamp(max=clip)

        q_return = compute_q_return(state_values, done_masks, rewards, rho_clamped, q_values_action,
                                    is_first_in_sequence)

        loss = compute_loss(q_return, q_values_action, state_values, policy, policy_action, rho, rho_clamped, clip,
                            q_values)

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
