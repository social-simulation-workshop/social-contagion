import argparse
import numpy as  np
import random
from plot import Plot2DArray
import os
# init
K = 6 # the number of cultural practices
N = 50 # the number of agents
decay_rate = 0.9 # the decay rate of mattrix R
np.random.seed(34)

class Agent:
    def __init__(self):
        self.R = np.ones( (K,K)  )# the R matrix (an K*K numpy array)
        self.V = np.random.uniform(low=-1, high=1, size=(1,K))
        self.V = self.V[0]
        #exit()
        self.P = np.empty( (1,K) ) # to store the agent's likelihood of exhibiting the K behaviors
        self.update_P()

    def update_P(self):
        eV = np.exp(self.V) # eV[i] = e^(V[i])
        sum_ev = np.sum(eV) # 機率分母的部分
        self.P = eV/sum_ev
        
    def calculate_CS(self):
        Omega = np.zeros((K, K))
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
        cs *= 1/(K*(K-1))
        return cs

class simulate:
    def __init__(self):
        self.agents = []
        self.map = []
        self.times = 100001 # times of iterations
        for _ in range(N):
            agent = Agent()
            self.agents.append(agent)
            self.map.append(agent.V)
        self.plotter = Plot2DArray() # visual

    def act(self,agent):
        # P(i) = e^(V_i) / ( sigma(j in K)  e^(v_j) )
        #print(agent.P)
        b1 = 0
        b2 = 0
        while b1 == b2:
            b1, b2 = np.random.choice(K, 2, p = agent.P)

        return b1, b2
    
    def Preference_Similarity(self):
        ans = 0
        for i in range(N):
            for j in range(i+1,N):
                r = np.corrcoef(self.agents[i],self.agents[j])
                ans += r[0,1]
        ans *= 2/(N*(N-1))
        return ans

    def Preference_Congruence(self):
        ans = 0
        for i in range(N):
            for j in range(i+1,N):
                r = np.corrcoef(self.agents[i],self.agents[j])
                ans += abs(r[0,1])
        ans *= 2/(N*(N-1))
        return ans

    def run(self):
        for time in range(self.times):
            # chose two people
            A,B = np.random.choice(N, 2)
            agent_A = self.agents[A]
            agent_B = self.agents[B]

            # act and observe
            b1, b2 = self.act(agent_A)
            agent_B.R[b1][b2] += 1
            agent_B.R[b2][b1] += 1
            
            # update V and decide whether to retain or not
            ori_cs = agent_B.calculate_CS()
            delta_v = np.random.normal(size=1)
            
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
            
            # R decays
            agent_B.R *= decay_rate
            '''
            # measurement
            ### cognitive agreement
            ##### interpretative distance
            group_dis = 0
            for i in range(N):
                for j in range(N):
                    dis = 0
                    for k in range(K):
                        for l in range(K):
                            dis += abs(self.agents[i].R[k][l]/np.max(self.agents[i].R) - self.agents[j].R[k][l]/np.max(self.agents[j].R))
                    dis /= (K**2)
                    group_dis += dis
            group_dis /= (N**2) # interpretative distance at the group level

            ### behavioral agreement
            ##### mutual information
            I = 0   # mutual information
            for x in range(K):
                p_x = 0 # P(b1 = x)
                for i in range(N):
                    p_x += self.agents[i].P[x]
                p_x /= N
                for y in range(K):
                    if y == x:  # P(b1 = x, b2 = x) = 0 given condition
                        continue
                    p_y = 0 # P(b2 = y)
                    p_x_y = 0 # P(b1 = x, b2 = y)
                    for i in range(N):
                        for j in range(K):  # enumerate X(b1) (using variable j) to get the marginal probability of y
                            if j == y:
                                continue
                            p_y += self.agents[i].P[j]*self.agents[i].P[y]/(1-self.agents[i].P[j])

                        p_x_y += self.agents[i].P[x]*self.agents[i].P[y]/(1-self.agents[i].P[x])
                    p_y /= N
                    p_x_y /= N
                    I += p_x_y*np.log2(p_x_y/p_x/p_y)
            '''
            # R decays
            agent_B.R *= decay_rate

            if time%10000 == 0:
                total = 0
                for agent in self.agents:
                    total += agent.calculate_CS()
                total /= N
                print(total)

            if time %1000 == 0:
                self.draw(time)

    def draw(self,time):    
        QQ = []
        now = 0
        QQ.append(self.map[now])
        last = [ i for i in range(1,N) ]
        for __ in range(N-1):
            counting = np.zeros(N)
            for _ in range(100):
                act1,act2 = self.act(self.agents[now])
                for k in last:
                    counting[k] += 1
                    if (act1,act2) == self.act(self.agents[k]):
                        counting[k] += 1 
            most_sim = np.argmax(counting)
            now = most_sim
            QQ.append(self.map[now])
            last.remove(now)
        self.plotter.plot_map( QQ, time)


if __name__ == "__main__":
    img_dir = os.path.join(os.getcwd(), 'imgfiles')
    demo = simulate()
    demo.run()
    demo.plotter.save_gif(fps=100)
    demo.plotter.save_mp4()