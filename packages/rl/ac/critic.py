from packages.rl.ac.net import Q_Net,PolicyNet


import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch


class Critic:
    def __init__(self,config) -> None:
    
        # conifg 
        self.config  = config 
        
        # netowrk config 
        self.net = Q_Net(config.state_dims)
        
        
        # optimizer and loss function
        self.optimizer = torch.optim.Adam(self.net.parameters(),config.value_lr)
        self.loss_func = nn.MSELoss()
        
    
    
    
    def get_output(self,x:np.ndarray)->np.ndarray:
        x = torch.FloatTensor(x)
        return self.net(x)
    
    
    def learn(self,v,v_,r):
        # minimize the td error
        loss = self.loss_func(r+self.config.gamma*v_,v)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss