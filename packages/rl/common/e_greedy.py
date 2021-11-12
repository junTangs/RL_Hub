import random 
import numpy as np
import torch

def select_action(e:float,action_values:np.ndarray)->int:
    """采用e-greedy策略选取动作

    Args:
        e (float): e值
        action_values (np.ndarray): 动作对应的value值 [a1_value,a2_value,...,an_value]
                                    shape:(action_dims,)

    Returns:
        [int]: 选取的动作的index值
    """
    rand = random.random()
    if rand <= e:
        return np.random.randint(action_values.shape[0])
    else:
        return np.argmax(action_values) 
    

def select_action_gpu(e:float,action_values:torch.Tensor)->int:
    """采用e-greedy策略选取动作

    Args:
        e (float):e值
        action_values (torch.Tensor): 动作对应的value值 [a1_value,a2_value,...,an_value]
                                    shape:(action_dims,)

    Returns:
        int: 选取的动作的index值
    """
    rand = random.random()
    if rand <= e:
        return np.random.randint(0,action_values.shape[0])
    else:
        return int(torch.argmax(action_values))
    
    
    
if __name__ == "__main__":
    
    action_values = np.array([0.1,22,3,4])
    a = select_action(0.0005,action_values)
    print(a)
    
    
    action_values = torch.Tensor(action_values)
    a = select_action_gpu(1,action_values)
    print(a)
    