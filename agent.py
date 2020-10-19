import numpy as np
import random
import pickle
import copy
import time
from collections import namedtuple, deque
import pandas as pd
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = 100000    # replay buffer size
BATCH_SIZE = 1024       # minibatch size
GAMMA = 0.99            # discount factor
LR = 0.00005            # learning rate
TAU = 0.001             # for soft update of target parameters

base_dir = './data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """ Interacts with and learns from the environment """

    def __init__(self, state_size = 4*4, action_size = 4, seed = 42,
                 fc1_units=256, fc2_units=256, fc3_units=256, 
                 buffer_size = BUFFER_SIZE, batch_size = BATCH_SIZE, 
                 lr = LR, use_expected_rewards = True, predict_steps = 2,
                 gamma = GAMMA, tau = TAU):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            fc*_units (int): size of the respective layer
            buffer_size (int): number of steps to save in replay buffer
            batch_size (int): self-explanatory
            lr (float): learning rate
            use_expected_rewards (bool): whether to predict the weighted sum of future rewards or just for current step
            predict_steps (int): for how many steps to predict the expected rewards
            
        """
        TAU = tau
        GAMMA = gamma
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.batch_size = batch_size
        self.losses = []
        self.use_expected_rewards = use_expected_rewards
        self.current_iteration = 0
        
        # Game scores
        self.scores_list = []
        self.last_n_scores = deque(maxlen=50)
        self.mean_scores = []
        self.max_score = 0
        self.min_score = 1000
        self.best_score_board = []

        # Rewards
        self.total_rewards_list = []
        self.last_n_total_rewards = deque(maxlen=50)
        self.mean_total_rewards = []
        self.max_total_reward = 0
        self.best_reward_board = []

        # Max cell value on game board
        self.max_vals_list = []
        self.last_n_vals = deque(maxlen=50)
        self.mean_vals = []
        self.max_val = 0
        self.best_val_board = []

        # Number of steps per episode
        self.max_steps_list = []
        self.last_n_steps = deque(maxlen=50)
        self.mean_steps = []
        self.max_steps = 0
        self.total_steps = 0
        self.best_steps_board = []
        
        self.actions_avg_list = []
        self.actions_deque = {
            0:deque(maxlen=50),
            1:deque(maxlen=50),
            2:deque(maxlen=50),
            3:deque(maxlen=50)
        }

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units, fc3_units = fc3_units).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units, fc3_units = fc3_units).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        lr_s = lambda epoch: 0.998 ** (epoch % 1000) if epoch < 100000 else 0.999 ** (epoch % 1000) 
        self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9999)

        # Replay buffer
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        # Initialize time step
        self.t_step = 0
        self.steps_ahead = predict_steps

    def save(self, name):
        """Saves the state of the model and stats
        
        Params
        ======
            name (str): name of the agent version used in dqn function
        """
        
        torch.save(self.qnetwork_local.state_dict(), base_dir+'/network_local_%s.pth' % name)
        torch.save(self.qnetwork_target.state_dict(), base_dir+'/network_target_%s.pth' % name)
        torch.save(self.optimizer.state_dict(), base_dir+'/optimizer_%s.pth' % name)
        torch.save(self.lr_decay.state_dict(), base_dir+'/lr_schd_%s.pth' % name)
        state = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'seed': self.seed,
            'batch_size': self.batch_size,
            'losses': self.losses,
            'use_expected_rewards': self.use_expected_rewards,
            'current_iteration': self.current_iteration,
        
        # Game scores
            'scores_list': self.scores_list,
            'last_n_scores': self.last_n_scores,
            'mean_scores': self.mean_scores,
            'max_score': self.max_score,
            'min_score': self.min_score,
            'best_score_board': self.best_score_board,

        # Rewards
            'total_rewards_list': self.total_rewards_list,
            'last_n_total_rewards': self.last_n_total_rewards,
            'mean_total_rewards': self.mean_total_rewards,
            'max_total_reward': self.max_total_reward,
            'best_reward_board': self.best_reward_board,

        # Max cell value on game board
            'max_vals_list': self.max_vals_list,
            'last_n_vals': self.last_n_vals,
            'mean_vals': self.mean_vals,
            'max_val': self.max_val,
            'best_val_board': self.best_val_board,

        # Number of steps per episode
            'max_steps_list': self.max_steps_list,
            'last_n_steps': self.last_n_steps,
            'mean_steps': self.mean_steps,
            'max_steps': self.max_steps,
            'total_steps': self.total_steps,
            'best_steps_board': self.best_steps_board,
        
            'actions_avg_list': self.actions_avg_list,
            'actions_deque': self.actions_deque,
        # Replay buffer
            'memory': self.memory.dump(),
        # Initialize time step
            't_step': self.t_step,
            'steps_ahead': self.steps_ahead
        }

        with open(base_dir+'/agent_state_%s.pkl' % name, 'wb') as f:
            pickle.dump(state, f)

    def load(self, name):
        """Saves the state of the model and stats
        
        Params
        ======
            name (str): name of the agent version used in dqn function
        """
        self.qnetwork_local.load_state_dict(torch.load(base_dir+'/network_local_%s.pth' % name))
        self.qnetwork_target.load_state_dict(torch.load(base_dir+'/network_target_%s.pth' % name))
        self.optimizer.load_state_dict(torch.load(base_dir + '/optimizer_%s.pth' % name))
        self.lr_decay.load_state_dict(torch.load(base_dir + '/lr_schd_%s.pth' % name))
        
        with open(base_dir+'/agent_state_%s.pkl' % name, 'rb') as f:
            state = pickle.load(f)

        self.state_size = state['state_size']
        self.action_size = state['action_size']
        self.seed = state['seed']
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.batch_size = state['batch_size']
        self.losses = state['losses']
        self.use_expected_rewards = state['use_expected_rewards']
        self.current_iteration = state['current_iteration']
        
        # Game scores
        self.scores_list = state['scores_list']
        self.last_n_scores = state['last_n_scores']
        self.mean_scores = state['mean_scores']
        self.max_score = state['max_score']
        self.min_score = state['min_score'] if 'min_score' in state.keys() else state['max_score']
        self.best_score_board = state['best_score_board']

        # Rewards
        self.total_rewards_list = state['total_rewards_list']
        self.last_n_total_rewards = state['last_n_total_rewards']
        self.mean_total_rewards = state['mean_total_rewards']
        self.max_total_reward = state['max_total_reward']
        self.best_reward_board = state['best_reward_board']

        # Max cell value on game board
        self.max_vals_list = state['max_vals_list']
        self.last_n_vals = state['last_n_vals']
        self.mean_vals = state['mean_vals']
        self.max_val = state['max_val']
        self.best_val_board = state['best_val_board']

        # Number of steps per episode
        self.max_steps_list = state['max_steps_list']
        self.last_n_steps = state['last_n_steps']
        self.mean_steps = state['mean_steps']
        self.max_steps = state['max_steps']
        self.total_steps = state['total_steps']
        self.best_steps_board = state['best_steps_board']
        
        self.actions_avg_list = state['actions_avg_list']
        self.actions_deque = state['actions_deque']

        # Replay buffer
        self.memory.load(state['memory'])
        
        # Initialize time step
        self.t_step = state['t_step']
        self.steps_ahead = state['steps_ahead']

        
    def step(self, state, action, reward, next_state, done, error, action_dist):
        # Save experience in replay memory    
        self.memory.add(state, action, reward, next_state, done, error, action_dist, None)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        return action_values.cpu().data.numpy()
        
    def learn(self, learn_iterations, mode = 'board_max', save_loss = True, gamma = GAMMA, weight = None):
        
        if self.use_expected_rewards:
            self.memory.calc_expected_rewards(self.steps_ahead, weight)
            
        self.memory.add_episode_experiences()
        
        losses = []
        
        if len(self.memory) > self.batch_size:
            if learn_iterations is None:
                learn_iterations = self.learn_iterations
        
            for i in range(learn_iterations):

                states, actions, rewards, next_states, dones = self.memory.sample(mode=mode)
                
                # Get expected Q values from local model
                Q_expected = self.qnetwork_local(states).gather(1, actions)
                

                # Compute loss
                loss = F.mse_loss(Q_expected, rewards)
                
                losses.append(loss.detach().numpy())

                # Minimize the loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.lr_decay.step()
            
            if save_loss:
                self.losses.append(np.mean(losses))
        else:
            self.losses.append(0)
                
    def soft_update(self, local_model, target_model, tau):
        """NOT USED ANYMORE
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        
        self.episode_memory = []
        self.batch_size = batch_size
        
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "error", "action_dist", "weight"])
        self.geomspaces = [np.geomspace(1., 0.5, i) for i in range(1, 10)]

    def dump(self):
        # Saves the buffer into dict object
        d = {
            'action_size': self.action_size,
             'batch_size': self.batch_size,
             'seed': self.seed,
             'geomspaces': self.geomspaces
        }

        d['memory'] = [d._asdict() for d in self.memory]
        return d

    def load(self, d):
        # creates a new buffer from dict
        self.action_size = d['action_size']
        self.batch_size = d['batch_size']
        self.seed = d['seed']
        self.geomspaces = d['geomspaces']

        for e in d['memory']:
            self.memory.append(self.experience(**e))

    def reset_episode_memory(self):
        self.episode_memory = []
        
    def add(self, state, action, reward, next_state, done, error, action_dist, weight = None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, error, action_dist, weight)
        self.episode_memory.append(e)

    def add_episode_experiences(self):
        self.memory.extend(self.episode_memory)
        self.reset_episode_memory()
        
    def calc_expected_rewards(self, steps_ahead = 1, weight = None):

        rewards = [e.reward for e in self.episode_memory if e is not None]

        exp_rewards = [np.sum(rewards[i:i+steps_ahead] * self.geomspaces[steps_ahead-1]) for i in range(len(rewards) - steps_ahead)]

        temp_memory = []
        
        for i, e in enumerate(self.episode_memory[:-steps_ahead]):
            t_e = self.experience(e.state, e.action, exp_rewards[i], e.next_state, e.done, e.error, e.action_dist, weight)
            temp_memory.append(t_e)

        self.episode_memory = temp_memory
            
    def sample(self, mode='board_max'):
        """Randomly sample a batch of experiences from memory."""
        
        if mode == 'random':
            experiences = random.sample(self.memory, k=self.batch_size)
        elif mode == 'board_max':
            probs = np.array([e.state.max() for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
                
        elif mode == 'board_sum':
            probs = np.array([e.state.sum() for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
                
        elif mode == 'reward':
            # Shifting by +1 is to keep steps with 0 reward in training set, otherwise they will receive 0 probability during sampling
            probs = np.array([e.reward + 1 for e in self.memory]) 
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
                
        elif mode == 'error':
            probs = np.array([e.error for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'error_u':
            probs = np.array([e.error for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'weighted_error':
            weights = np.array([e.weight for e in self.memory])
            max_weight = weights.max()
            sum_weight = weights.sum()
            weights = np.array([(max_weight - w + 1) / (sum_weight + len(weights)) for w in weights])
            probs = np.array([e.error for e in self.memory])
            probs = probs * weights
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'weighted_error_reversed':
            weights = np.array([e.weight for e in self.memory])
            sum_weight = weights.sum()
            weights = np.array([(w) / (sum_weight) for w in weights])
            probs = np.array([e.error for e in self.memory])
            probs = probs * weights
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'action_balanced_error':
            probs = np.array([e.error * e.action_dist for e in self.memory])
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), size=self.batch_size, p=probs)
            experiences = deque(maxlen=self.batch_size)        
            for i in idx:
                experiences.append(self.memory[i])
        
        elif mode == 'clipped_error':
            t = pd.DataFrame(self.memory)
            
            t = t[t['error'] < t['error'].quantile(0.99)]
            t['probs'] = t['error'] * t['action_dist']
            t['probs'] = t['probs'] / t['probs'].sum()
            idx = np.random.choice(len(t), size=self.batch_size, p=t['probs'].values)
            t = t.iloc[idx]
            
            experiences = deque(maxlen=self.batch_size)        
            for i in list(t.itertuples(name='Experience', index=False)):
                experiences.append(i)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)