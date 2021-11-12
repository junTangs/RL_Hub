

import torch
from torch._C import set_flush_denormal
from packages.rl.ac.actor import Actor
from packages.rl.ac.critic import Critic
from torch.distributions import Categorical
from torch.autograd import Variable

from packages.rl.common.replay_memory import ReplayMemory

import  numpy as np

class ACConfig:
    def __init__(self) -> None:
        # network config
        self.state_dims = 10
        self.action_dims = 10
        
        self.policy_lr = 1e-3
        self.value_lr = 1e-2
        
        # other config
        self.gamma = 0.9
        
        # replay buffer config 
        self.batch_size = 32
        self.replay_buffer_size = 2000
        
        
        
        
class ActorCritic:
    def __init__(self,config:ACConfig) -> None:
        self.config = config
        
        # config actor and critic 
        self.actor = Actor(config)
        self.critic = Critic(config)
        
        # config replay buffer 
        self.replay_buffer = ReplayMemory(config.replay_buffer_size)
        
        # counter
        self.learn_counter = 0
        
    def choose_action(self,x:np.ndarray):
        """从输入状态选择一个动作

        Args:
            x (np.ndarray): 输入状态

        Returns:
            int: 输出的动作的标签
        """
        return self.actor.choose_action(x)
    
    
    def store_transition(self,s,a,r,s_):
        self.replay_buffer.push(*s,a,r,*s_)
        return 
        
        
    def calculate_td_error(self,s,s_,r):
        # get v(s) and v(s') and td-error
        state_value = self.critic.get_output(s)
        next_state_value = self.critic.get_output(s_)
        
        # calculate td-error
        td_error = r+ self.config.gamma*next_state_value-state_value
        return td_error,state_value,next_state_value
    
    def learn(self):
        
         # sample from memory 
        
        sample_batch = self.replay_buffer.sample(self.config.batch_size)
        
        s = Variable(torch.FloatTensor(sample_batch[:,:self.config.state_dims]))
        a = Variable(torch.LongTensor(sample_batch[:,self.config.state_dims:self.config.state_dims+1]))
        r = Variable(torch.FloatTensor(sample_batch[:,self.config.state_dims+1:self.config.state_dims+2]))
        s_ = Variable(torch.FloatTensor(sample_batch[:,-self.config.state_dims:]))
        
        td_error,v,v_= self.calculate_td_error(s,s_,r)
        
        v = Variable(v,requires_grad=True)
        v_ = Variable(v_,requires_grad=True)
        
        loss_a = self.actor.learn(s,a,r,s_,td_error)
        loss_c = self.critic.learn(v,v_,r)
        
        
        return float(loss_a),float(loss_c)