import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from SumTree import SumTree

BUFFER_SIZE = int(1e5)  #size of replay buffer
BATCH_SIZE = 64 # size of sample mini batches        
GAMMA = 0.99 # Discount rate
PRI_A = 0.05 # priority experience replay (PER) coefficient 'a'
PRI_EPSILON = 1e-2 # PER coefficient epsilon 
TAU = 1.5e-3 # used for soft update          
LR = 1.5e-4 #the learning rate used by all four NNs. 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    # Initializes the agent and all four NNs.
    def __init__(self, state_size, action_size, seed):
        
        # Define state and action space size and random seed used in NNS
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Define the actor NNs
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)

        # Define the critic NNs
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR)

        #Define the replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
    
    #Called every timestep t in order to learn
    def step(self, state, action, reward, next_state, done):        
        self.memory.add(state, action, reward, next_state, done)
        
        # Tell the agent to learn when there are enough experiences in the buffer in order to create a mini batch
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
    
    # Chooses an action for the agent to perform given a particular state
    def act(self, state):
        # Choose the best action
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(torch.from_numpy(state).float().to(device)).cpu().data.numpy() 
        self.actor_local.train()
        
        return np.clip(action, -1, 1)

    # Updates the parameters of all four NNs every timestep t
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences # Get experiences from the replay buffer
       
        # updates weights for critic network
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
 
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets) # Calculates critic loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # updates weights for actor network
        actions_actor = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_actor).mean() # Calculates actor loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # applies weights to NNs using a soft update rule
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    # Updates weights in NN using soft update rule
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size # action size of agent
        self.memory = deque(maxlen=buffer_size)  # erplay buffer
        self.batch_size = batch_size # size of mini batches
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"]) #experience to be stores in buffer
        self.seed = random.seed(seed) # random seed used for sampling
    
    def add(self, state, action, reward, next_state, done):
        #adds experience to replay buffer
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
    
    def sample(self):
        #samples k experience from buffer
        experiences = random.sample(self.memory, k=self.batch_size)

        #separates experience into SARS' lists
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        #returns length of buffer
        return len(self.memory)