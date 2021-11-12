import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import numpy as np
from packages.rl.common.replay_memory import ReplayMemory
from packages.rl.common.e_greedy import select_action_gpu
from packages.rl.dqn.net import Q_Net


class DQNConfig:
    def __init__(self):
        # replay buffer config
        self.replay_buffer_size = 2000
        self.batch_size = 32
        
        # network config
        self.state_dims = 10
        self.action_dims = 10
        
        self.lr = 1e-3
        
        # other config
        self.e = 1e-2
        self.learn_freq = 100
        self.gamma = 10 
        
        

class DQN:
    def __init__(self,config:DQNConfig) -> None:
        """ Nature DQN

        Args:
            config (DQNConfig): DQN设置实例
        """
        self.config = config
        self.replay_buffer  = ReplayMemory(config.replay_buffer_size)
        
        self.target_net = Q_Net(config.state_dims,config.action_dims)
        self.eval_net = Q_Net(config.state_dims,config.action_dims)
        
        # counter
        self.learn_counter = 0
        
        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),config.lr)
        self.loss_func = nn.MSELoss()
        
        # other
        self.e = config.e
        
        return 
    
    def choose_action(self,x:np.ndarray)->int:
        """从输入状态选择一个动作

        Args:
            x (np.ndarray): 输入状态

        Returns:
            int: 输出的动作的标签
        """
        # (state_dims,)
        x = torch.FloatTensor(x)
        
        action_values = self.eval_net(x)
        
        action = select_action_gpu(self.e,action_values)
        
        return action
    
    
    def store_transition(self,s,a,r,s_):
        
        self.replay_buffer.push(*s,a,r,*s_)
        
        return 
    
    def get_q_target(self,s,a,r,s_):
        """获取目标Q值

        Args:
            s ([type]): 状态
            a ([type]): 动作
            r ([type]): 奖励
            s_ ([type]): 次态

        Returns:
            torch.Tensor: q_target的值
        """
        q_next  = self.target_net(s_).detach()
        q_target = r + self.config.gamma * q_next.max(1)[0].view(self.config.batch_size,1)
        return q_target
    
    def learn(self):
        if self.learn_counter%self.config.learn_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.learn_counter+= 1
        
        # sample from memory 
        
        sample_batch = self.replay_buffer.sample(self.config.batch_size)
        
        s = Variable(torch.FloatTensor(sample_batch[:,:self.config.state_dims]))
        a = Variable(torch.LongTensor(sample_batch[:,self.config.state_dims:self.config.state_dims+1]))
        r = Variable(torch.FloatTensor(sample_batch[:,self.config.state_dims+1:self.config.state_dims+2]))
        s_ = Variable(torch.FloatTensor(sample_batch[:,-self.config.state_dims:]))
        
        
        # eval net 
        
        q_eval = self.eval_net(s).gather(1,a)
        
        q_target = self.get_q_target(s,a,r,s_)
        
        loss = self.loss_func(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return float(loss)
        
        
class DoubleDQN(DQN):
    def __init__(self, config: DQNConfig) -> None:
        """DoubleDQN

        Args:
            config (DQNConfig): DQN设置实例
        """
        super().__init__(config)
    
    
    def get_q_target(self,s,a,r,s_):
        
        q_eval_next = self.eval_net(s_).detach() #(batch_size,action_dims)
        eval_next_action  = torch.argmax(q_eval_next,1) #(batch_size,)
        
        q_next = self.target_net(s_).detach() #(batch_size,action_dims)
        
        term2 = self.config.gamma*q_next[:,eval_next_action][:,0].view(self.config.batch_size,1)

        return r+term2
    
    