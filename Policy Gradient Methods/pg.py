from utils import *


class Policy(nn.Module):
    """
    The network for the naive policy gradient method.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.0002, gamma=0.98):
        super(Policy, self).__init__()
        self.data = []
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        """
        Forward pass of the neural network.
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=0)
        return x

    def start_training(self):
        """
        Train the neural network using the stored data.
        """
        # Compute discounted rewards for the stored data
        rewards, probs = zip(*self.data)
        discounted_rewards = compute_discounted_rewards(rewards, self.gamma)

        # Zero the gradients before backpropagation
        self.optimizer.zero_grad()

        # Compute and accumulate the loss for each data point
        total_loss = 0
        for prob, discounted_reward in zip(probs, discounted_rewards):
            loss = -torch.log(prob) * discounted_reward
            total_loss += loss

        # Perform backpropagation and optimization step
        total_loss.backward()
        self.optimizer.step()

        # Clear stored data for the next training iteration
        self.data = []
