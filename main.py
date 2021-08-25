import argparse
import numpy as  np
import random

# init
K = 1000 # the size of the set of cultural practices
N = 1000 # the number of agents
decay_rate = 0.3 # the  decay rate of mattrix R

class agent:
    def __init__(self):
        self.R = np.ones(K,K) # the R matrix (an K*K numpy array)
        self.V = np.random.uniform(low=0,high=1,size=(K,1)) # ?????????????? 左開右避？

class simulate:
    def __init__(self):
        self.agents = []
        self.times = 100000 # how many times to iterate
        for _ in range(N):
            self.agents.append(agent())

    def act(self,agent):
        # P(i) = e^(V_i) / ( sigma(j in K)  e^(v_j) )
        pass
    
    def run(self):
        for time in range(self.times):
            # chose two people
            A,B = ("random","two number")
            agent_A = self.agents[A]
            agent_B = self.agents[B]
            # act and observe
            b1 = self.act(agent_A)
            b2 = self.act(agent_A)
            agent_B.R[b1][b2] += 1
            agent_B.R[b2][b1] += 1
            
            # update or not?
            # concordance between V and R


# start simulateing
