
import torch
import torch.nn as nn
import torch.nn.functional as F




class Q_Net(nn.Module):
    def __init__(self,state_dims:int):
        """深度Q网络

        Args:
            state_dims (int): 输入状态的维度
            action_dims (int): 输出动作的维度
        """
        super(Q_Net,self).__init__()
        
        # define network structure
        self.fc1 = nn.Linear(state_dims,8)
        self.fc2 = nn.Linear(8,1)

        
        return 
    
    def forward(self,x)->torch.Tensor:
        """深度Q网络的前向过程

        Args:
            x (troch.Tensor): 输入的状态值 shape:(states_dims,)

        Returns:
            [torch.Tensor]: 每个动作的value值 shape:(action_dims,)
        """
        x = self.fc1(x)
        x = F.relu(x)
        
        
        x = self.fc2(x)
        x = F.relu(x)
        
        return x
    
        
    


class PolicyNet(nn.Module):
    def __init__(self,state_dims:int,action_dims:int):
        """策略网络

        Args:
            state_dims (int): 输入状态的维度
            action_dims (int): 输出动作的维度
        """
        super(PolicyNet,self).__init__()
        
        # define network structure
        self.fc1 = nn.Linear(state_dims,8)
        self.fc2 = nn.Linear(8,16)
        self.fc3 = nn.Linear(16,8)
        self.out = nn.Linear(8,action_dims)
        
        return 
    
    def forward(self,x)->torch.Tensor:
        """深度Q网络的前向过程

        Args:
            x (troch.Tensor): 输入的状态值 shape:(states_dims,)

        Returns:
            [torch.Tensor]: 每个动作的value值 shape:(action_dims,)
        """
        x = self.fc1(x)
        x = F.relu(x)
        
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        return self.out(x) 