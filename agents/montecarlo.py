import numpy as np

class MonteCarloAgent():

    def __init__(self, action_space, state_space, epsilon=0.9, discount=0.95) -> None:
        self.action_space =  action_space
        self.state_space =  state_space
        self.states = []
        self.actions = []
        self.rewards = []
        self.discount = discount
        self.epsilon = epsilon
        self.Q_s = {state:-1 for state in self.state_space}

    def update(self):
        """
        Input: 
            states (list): states of an episode generated from generate_episode
            actions (list): actions of an episode generated from generate_episode
            rewards (list): rewards of an episode generated from generate_episode
            discount (float): discount factor
        Returns visited_states_returns (dictionary): 
            keys are all the unique state-action combinations in the episode
            values are the estimated discounted return of the first visited pair
        """
        visited_states_returns = {}
        # TO IMPLEMENT
        # --------------------------------
        G = 0
        T = len(self.states)
        for t in range(T-1, 0, -1):
            G = self.discount * G + self.rewards[t]
            # if self.states[t] not in self.states[:t]:
            visited_states_returns[self.states[t]] = visited_states_returns.get(self.states[t],[]) + [G]

        # --------------------------------
        visited_states_returns = {state:np.mean(visited_states_returns.get(state,[0]))*0.5+self.Q_s[state]*0.5 for state in visited_states_returns}
        self.Q_s.update(visited_states_returns)

    def act(self, env, greedy=False):
        curr_state = env.agent_position
        next_state_values = []
        for action in self.action_space:
            res = env.step(action, reset_position=True)
            next_state_values.append(self.Q_s[res[0][0]])
        best_action = self.action_space[np.array(next_state_values).argmax()]
        if np.random.random()>self.epsilon and not greedy:
            return np.random.choice(self.action_space)
        else:
            return best_action            
    
    def log(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)        
        self.rewards.append(reward)