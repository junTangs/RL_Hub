
import random
import  collections
import numpy as np

class ReplayMemory(list):
    def  __init__(self,capacity:int) -> None:
        """初始化一个Replay Buffer

        Args:
            capacity (int): 经验池的容量
        """
        list.__init__(self)
        self.capacity = capacity
        self.position = 0
        
        return 
    
    def push(self,*args):
        """向经验池里推入一个(s,a,r,s_)
        Example:
            replay_buffer.push(1,2,2,3)
        """
        if len(self)< self.capacity:
            self.append(None)
        self[self.position] = list(args)
        # ringbuffer storage
        self.position = (self.position+1)%self.capacity
        
    
    def sample(self,batch_size:int)->np.ndarray:
        """从经验池中采样batch_size个Tansition

        Args:
            batch_size ([int]): batch_size的大小

        Returns:
            [np.ndarray]: 采样的[s,a,r,s_]组成的ndarrayu
        """
        return np.array(random.sample(self,batch_size))
    
    
if __name__ == "__main__":
    replay_buffer = ReplayMemory(10)
    replay_buffer.push(1,1,2,2)
    replay_buffer.push(1,2,2,2)
    replay_buffer.push(1,3,2,2)
    sampled = replay_buffer.sample(2)
    print(sampled.shape)
    print(len(replay_buffer))
    