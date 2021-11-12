from packages.rl.ac.net import Q_Net,PolicyNet


import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.distributions import Categorical

class Actor:
    def __init__(self,config) -> None:
    
        # conifg 
        self.config  = config 
        
        # netowrk config 
        self.policy_net = PolicyNet(config.state_dims,config.action_dims)
        
        
        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),config.policy_lr)
        self.loss_fn = nn.SmoothL1Loss()
    
    def choose_action(self,x:np.ndarray)->int:
        """从输入状态选择一个动作

        Args:
            x (np.ndarray): 输入状态

        Returns:
            int: 输出的动作的标签
        """
        x = torch.FloatTensor(x)
        
        action_pred = self.policy_net(x)
        
        action_prob = F.softmax(action_pred,dim=-1)
        action_distribution = Categorical(action_prob)
        
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action).unsqueeze(0)
        

        
        return int(action),log_prob
    
    def get_output(self,x:np.ndarray):
        x = torch.FloatTensor(x)
        return self.policy_net(x)
    
    def learn(self,s,a,r,s_,td_error):
        
        action_pred= self.policy_net(s)
        action_prob = F.softmax(action_pred,dim=-1)
        
        action_distribution = Categorical(action_prob)
        log_prob = action_distribution.log_prob(a).unsqueeze(0)

        # calculate loss
        loss = -(log_prob*td_error).mean() # minimize log_prob*td_error
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
        
        
        