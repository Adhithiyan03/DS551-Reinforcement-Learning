import random
import numpy as np
from collections import deque
from collections import namedtuple
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd 
from agent import Agent
from dqn_model import DQN
import time
import os

mem = deque(maxlen=500) 
mem_model = 100

gamma = 0.99
epilson = 0.02
ep_s = epilson
ep_end = 0.005
decay = 1000
batch_size = 32
alpha = 1e-5
target_update = 20000

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
        
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

LOAD = True


class Agent_Dueling_DQN():
    def __init__(self, env, args):
        self.env = env
        state = env.reset().transpose(2,0,1)
        self.target = Dueling_DQN(state.shape, self.env.action_space.n) 
        self.policy = Dueling_DQN(state.shape, self.env.action_space.n) 
        
        self.target.load_state_dict(self.policy_net.state_dict())
        self.policy = self.policy_net.cuda()
        self.targett = self.target_net.cuda()
        print('Network Loaded')
        
        if args.test_dqn or LOAD == True:
            print('loading trained model')
            checkpoint = torch.load('trainData')
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        
        self.target.load_state_dict(self.policy.state_dict())
            
    def init_game_setting(self):
        print('loading trained model')
        checkpoint = torch.load('trainData')
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])    
    
    def push(self, state, action, reward, next_state, done):
        state, next_state = np.expand_dims(state, 0), np.expand_dims(next_state, 0)
        mem.append((state, action, reward, next_state, done))
    
    def replay_buffer(self):
        state, action, reward, next_state, done = zip(*random.sample(mem, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)
    
    def make_action(self, observation, test=True):
        observation = observation.transpose(2,0,1)
        if np.random.random() > EPSILON or test==True:
            observation   = Variable(torch.FloatTensor(np.float32(observation)).unsqueeze(0), volatile=True)
            q_value = self.policy_net.forward(observation)
            action  = q_value.max(1)[1].data[0]
            action = int(action.item())            
        else:
            action = random.randrange(4)
        return action

    def optimize_model(self):
        state, action, next_state, reward, done  = self.replay_buffer()
        variable_state = Variable(torch.FloatTensor(np.float32(state)))
        variable_next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        variable_action = Variable(torch.LongTensor(action))
        variable_reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        state_action_value = self.policy_net(variable_state).gather(1, variable_actions.unsqueeze(1)).squeeze(1)
        next_state_value = self.target(variable_next_state).max(1)[0]
        expected_q_value = variable_reward + next_state_value * gamma * (1 - done) 

        loss = (state_action_values - Variable(expected_q_value.data)).pow(2).mean()
        return loss
        
        
    def train(self):
        ep = 0
        mean= 0
        avgr = []
        all_score = []
        step = 1
        store_epilson = []
        optimizer = optim.Adam(self.policy_net.parameters(), lr=ALPHA)
        while mean < 50:
                     
            state = self.env.reset()
            done = False
            ep_score = 0
            start = time.time()
            done = False

            while not done:

                action = self.make_action(state)    
                next_state, reward, done, _ = self.env.step(action)
                self.push(state.transpose(2,0,1), action, next_state.transpose(2,0,1), reward, done)
                state = next_state   
                
                if len(mem) > mem_model:
                    loss = self.optimize_model()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    ep = 0
                    continue        

                # Update exploration factor
                epilson = ep_end + (ep_s - ep_end) * math.exp(-1. * step/decay)
                store_epsilon.append(epsilon)
                step += 1
                
                ep_score += reward

                if step % target_update == 0:
                    self.target.load_state_dict(self.policy.state_dict())

            ep += 1
            all_score.append(ep_score)
            mean = np.mean(score[-100:])
            avgr.append(mean)
            
            if len(memory) > mem_model: 
                print('Episode: ', ep, ' score:', ep_score, ' Avg Score:',mean,' epsilon: ', epilson, ' t: ', time.time()-start, ' loss:', loss.item())

            if ep % 500 == 0:
                torch.save({
                    'epoch': ep,
                    'model_state_dict': self.policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'AvgRewards': avgr
                }, 'trainData')
                with open('Rewards.csv', mode='w', newline='') as dataFile:
                    csv.writer(dataFile).writerow(avgr)

        torch.save({
            'epoch': ep,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'AvgRewards': avgr
        }, 'trainData')
        with open('Rewards.csv', mode='w', newline='') as dataFile:
            csv.writer(dataFile).writerow(avgr)
