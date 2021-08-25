import argparse
import numpy as  np
import random

# init
K = 1000 # the number of cultural practices
N = 1000 # the number of agents
decay_rate = 0.3 # the decay rate of mattrix R

class agent:
    def __init__(self):
        self.R = np.ones(K,K) # the R matrix (an K*K numpy array)
        self.V = np.random.uniform(low=-1, high=1, size=(K,1))
        self.P = np.empty(K) # to store the agent's likelihood of exhibiting the K behaviors
        self.update_P()

    def update_P(self):
        eV = np.exp(self.V) # eV[i] = e^(V[i])
        sum_ev = np.sum(eV) # 機率分母的部分
        self.P = eV/sum_ev
    
    def calculate_CS(self):
        Omega = np.zeros(K, K)
        for i in range(K):
            for j in range(i+1, K):
                Omega[i][j] = abs(self.V[i]-self.V[j])
                Omega[j][i] = Omega[i][j]
        Omega /= np.max(Omega)
        standarized_R = self.R/np.max(self.R)
        cs = 0
        for i in range(K):
            for j in range(K):
                cs += abs(standarized_R[i][j]-Omega[i][j])
        cs = cs*K/(K*(K-1))
        return cs

class simulate:
    def __init__(self):
        self.agents = []
        self.times = 100000 # times of iterations
        for _ in range(N):
            self.agents.append(agent())

    def act(self,agent):
        # P(i) = e^(V_i) / ( sigma(j in K)  e^(v_j) )
        b1, b2 = np.random.choice(K, 2, p = agent.P)
        return b1, b2
    
    def run(self):
        for time in range(self.times):
            # chose two people
            A,B = np.random.randint(N, size=2)
            while A == B:
                A, B = np.random.randint(N, size=2)
            agent_A = self.agents[A]
            agent_B = self.agents[B]

            # act and observe
            b1, b2 = self.act(agent_A)
            agent_B.R[b1][b2] += 1
            agent_B.R[b2][b1] += 1
            
            # R decays
            R *= (1-decay_rate)

            # update V and decide whether to retain or not
            ori_cs = agent_B.calculate_CS()
            delta_v = np.random.normal(1)
            mean = np.mean(agent_B.V)
            if abs(agent_B.V[b1]-mean) <= abs(agent_B.V[b2]-mean):
                weaker = b1
            else:
                weaker = b2
            agent_B.V[weaker] += delta_v
            new_cs = agent_B.calculate_CS()
            if new_cs > ori_cs:
                agent_B.update_P()
            else:
                agent_B.V[weaker] -= delta_v

            # concordance between V and R


# start simulateing
