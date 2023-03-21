import gym
import numpy as np
from gym import spaces

class TwoDGridWorld(gym.Env):
    """
        - a size x size grid world which agent can ba at any cell other than terminal cell
        - terminal cell is set to be the last cell or bottom right cell in the grid world
        - 5x5 grid world example where X is the agent location and O is the tremial cell
          .....
          .....
          ..X..
          .....
          ....O -> this is the terminal cell where this is agent headed to  
        - Reference : https://github.com/openai/gym/blob/master/gym/core.py
    """
    metadata = {'render.modes': ['console']}
    
    # actions available 
    UP   = 0
    LEFT = 1
    DOWN = 2
    RIGHT= 3
    
    def __init__(self, size=4):
        super(TwoDGridWorld, self).__init__()
        
        self.size      = size # size of the grid world
        self.end_state = (size*size) - 1 # bottom right or last cell
        
        # randomly assign the inital location of agent
        # self.agent_position = np.random.randint( (self.size*self.size) - 1 )
        self.agent_position = 0
        
        # respective actions of agents : up, down, left and right
        self.action_space = spaces.Discrete(4)
        
        # set the observation space to (1,) to represent agent position in the grid world 
        # staring from [0,size*size)
        self.observation_space = spaces.Box(low=0, high=size*size, shape=(1,), dtype=np.uint8)

    def step(self,action, reset_position=False):
        info = {} # additional information
        
        reward = 0;
        init_position = self.agent_position
        row  = self.agent_position // self.size
        col  = self.agent_position % self.size
        if action == self.UP:
            if row != 0:
                self.agent_position -= self.size
            else:
                reward = -1
        elif action == self.LEFT:
            if col != 0:
                self.agent_position -= 1
            else:
                reward = -1
        elif action == self.DOWN:
            if row != self.size - 1:
                self.agent_position += self.size
            else:
                reward = -1
        elif action == self.RIGHT:
            if col != self.size - 1:
                self.agent_position += 1
            else:
                reward = -1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
        
        done   = bool(self.agent_position == self.end_state)
        reward = 1 if done else reward         
        new_position = self.agent_position
        if reset_position:
            self.agent_position = init_position
        # reward agent when it is in the terminal cell, else reward = 0
        
        return np.array([new_position]).astype(np.uint8), reward, done, None, info
    
    def render(self, mode='console'):
        '''
            render the state
        '''
        if mode != 'console':
          raise NotImplementedError()
        
        row  = self.agent_position // self.size
        col  = self.agent_position % self.size
        
        for r in range(self.size):
            for c in range(self.size):
                if r == row and c == col:
                    print("X",end='')
                else:
                    print('.',end='')
            print('')

    def reset(self):
        # -1 to ensure agent inital position will not be at the end state
        # self.agent_position = np.random.randint( (self.size*self.size) - 1 )
        self.agent_position = 0
        
        return np.array([self.agent_position]).astype(np.uint8)
    
    def close(self):
        pass
