import gym
from packages.rl.dqn.dqn import DQN,DQNConfig, DoubleDQN
from packages.common.utils_func import get_time_str,obj2dict


import json
import torch
import os

env = gym.make("CartPole-v0")
env = env.unwrapped

action_dims = env.action_space.n
state_dims = env.observation_space.shape[0]

print(action_dims,state_dims)

class MyDQNConfig(DQNConfig):
    def __init__(self):
        super().__init__()

        self.batch_size = 64
        self.action_dims = action_dims
        self.state_dims = state_dims
        self.gamma = 0.99
        self.e = 0.2
        self.lr = 0.01
        self.replay_buffer_size = 4000
        self.learn_freq = 100
    
dqn_config = MyDQNConfig()
dqn = DoubleDQN(dqn_config)



print("environment build success,start collection...")
for i_episode in range(500):
    
    print(f"episode:{i_episode}")
    s = env.reset()
    ep_r = 0
    
    while True:
        env.render()
        a = dqn.choose_action(s)
        
       
        s_,r,done,info = env.step(a)
        
        x,x_dot,theta,theta_dot = s_
        r1 = (env.x_threshold-abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians
        
        

        r = r1+r2
        ep_r+=r
        
        dqn.store_transition(s,a,r,s_)
        
        if len(dqn.replay_buffer) >= dqn.config.batch_size:
            dqn.learn()
            if done:
                print(f"episode:{i_episode}| reward:{ep_r} ")
        
        if done:
            break
        s = s_
        
        

time_str = get_time_str()
save_dir = f"trained_model/{time_str}_ddqn/"

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

config_file = open(os.path.join(save_dir,"config.json"),"w")
s = json.dumps(obj2dict(dqn_config))
config_file.write(s)
config_file.close()

torch.save(dqn.target_net.state_dict(),os.path.join(save_dir,"target_net.pth"))
torch.save(dqn.eval_net.state_dict(),os.path.join(save_dir,"eval_net.pth"))



    


        



