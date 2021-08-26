from matplotlib.pyplot import plot
import numpy as  np
import random
from plot import Plot2DArray
import os

np.random.seed(5)

class Agent:
    def __init__(self,K):
        '''
        K : a number
            the size of the set of the culture practices
        R : an K*K numpy array
            record the relate strength of culture practice i and j (for all i j in K)
        V : a Vector with length K
            to store the agents' likelihood of exhibiting the K behaviors
        P : a Vector with length K
            agents' probability of exhibiting the K behaviors (according to the vector V )
        '''
        self.K = K
        self.R = np.ones( (self.K, self.K) )
        self.V = np.random.uniform(low=-1, high=1, size=(1, self.K))
        self.V = self.V[0]

        self.P = np.empty( (1, self.K) ) 
        self.update_P() # use V to calculate the Probability

    def update_P(self):
        eV = np.exp(self.V) # eV[i] = e^(V[i])
        sum_ev = np.sum(eV) # 機率分母的部分
        self.P = eV/sum_ev
        
    def calculate_CS(self):
        # calculate the Constraint satisfaction
        Omega = np.zeros( (self.K, self.K) )
        for i in range( self.K ):
            for j in range(i+1, self.K):
                Omega[i][j] = abs(self.V[i]-self.V[j])
                Omega[j][i] = Omega[i][j]
        Omega /= np.max(Omega)
        standarized_R = self.R/np.max(self.R)
        cs = 0
        for i in range(self.K):
            for j in range(self.K):
                cs += abs(standarized_R[i][j]-Omega[i][j])
        cs *= 1/(self.K*(self.K-1))
        return cs

class simulate:
    def __init__(self):
        
        self.K = 6  # a number, the size of the set of the culture practices
        self.N = 30 # a number, the number of the agents
        self.decay_rate = 0.9 # the decay rate of mattrix R
        self.times = 100001 # the number of iterations
        
        # agents : agent list (init with N agent)
        self.agents = [ Agent(self.K) for _ in range(self.N) ]
        
        # init the obejects for plotting
        self.plot_init()

    def plot_init(self):
        tittle = [ "interpretative_distance","mutual_information","Preference_Similarity","Preference_Congruence"]
        self.plotter = Plot2DArray() # visual
        self.plot_img = [ Plot2DArray(filename_prefix = tittle[i] ) for i in range(4) ]
        self.plot_interpretative_distance = [[],[]]
        self.plot_mutual_information = [[],[]]
        self.plot_Preference_Similarity = [[],[]]
        self.plot_Preference_Congruence = [[],[]]
        self.plot_cluster_estimate = [[],[]]
        self.plot_period = 200
        #self.expectation_cluster_w(100)

    def act(self,agent):
        # Pick two practice with the agent's preference
        # the probability formula:
        #   P(i) = e^(V_i) / ( sigma(j in K)  e^(v_j) )
        b1, b2 = 0, 0
        while b1 == b2 : # b1, b2 can not be the same
            b1, b2 = np.random.choice(self.K, 2, p = agent.P)
        return b1, b2
    
    def Preference_Similarity(self,time):
        # measure the Preference_Similarity
        ans = 0
        for i in range(self.N):
            for j in range(i+1,self.N):
                r = np.corrcoef(self.agents[i].V,self.agents[j].V)
                ans += r[0,1]
        ans *= 2/(self.N*(self.N-1))
        # record the answer for the final plotting
        self.plot_Preference_Similarity[0].append(time)
        self.plot_Preference_Similarity[1].append(ans)

    def Preference_Congruence(self,time):
        # measure the Preference_Congruence
        ans = 0
        for i in range(self.N):
            for j in range(i+1,self.N):
                r = np.corrcoef(self.agents[i].V,self.agents[j].V)
                ans += abs(r[0,1])
        ans *= 2/(self.N*(self.N-1))
        # record the answer for the final plotting
        self.plot_Preference_Congruence[0].append(time)
        self.plot_Preference_Congruence[1].append(ans)

    def interpretative_distance(self,time):
        # measure the interpretative_distance
        group_dis = 0
        for agent_i in self.agents:     # calculate the group distance
            for agent_j in self.agents: # from each agent pair
                dis = 0
                for k in range(self.K):
                    for l in range(self.K): # distence between an agent pair's mattrix R
                        dis += abs(
                            agent_i.R[k][l] / np.max(agent_i.R)
                            - agent_j.R[k][l] / np.max(agent_j.R)
                        )
                group_dis += dis/(self.K**2) # Sum the distance
        group_dis /= (self.N**2) # interpretative distance at the group level
        # record the answer for the final plotting
        self.plot_interpretative_distance[0].append(time)
        self.plot_interpretative_distance[1].append(group_dis)
    
    def mutual_information(self,time):
        # measure the mutual information
        I = 0
        for x in range(self.K):
            p_x = 0 # the variable for P(b1 = x)
            for agent in self.agents:
                p_x += agent.P[x]
            p_x /= self.N
            for y in range(self.K):
                if y == x:  # P(b1 = x, b2 = x) = 0 given condition
                    continue
                p_y = 0 # the variable for P(b2 = y)
                p_x_y = 0 # the variable for P(b1 = x, b2 = y)
                for agent in self.agents:
                    # enumerate X(b1) (using variable j) to get the marginal probability of y
                    for j in range(self.K): 
                        if j == y: continue
                        p_y +=  agent.P[j] * agent.P[y]/( 1 - agent.P[j])
                    p_x_y += agent.P[x]*agent.P[y]/(1-agent.P[x])
                p_y /= self.N
                p_x_y /= self.N
                I += p_x_y*np.log(p_x_y/p_x/p_y)
        # record the answer for the final plotting
        self.plot_mutual_information[0].append(time)
        self.plot_mutual_information[1].append(I)
    
    def measurement(self,time):
        # call all the measurement
        self.Preference_Similarity(time)
        self.Preference_Congruence(time)
        self.interpretative_distance(time)
        self.mutual_information(time)

    def run(self):
        # the main function for simulation
        for time in range(self.times):
            # chose two different agent
            A,B = 0,0
            while A == B:
                A,B = np.random.choice(self.N, 2)  
            agent_A, agent_B = self.agents[A],self.agents[B]

            # act and observe
            b1, b2 = self.act(agent_A)
            agent_B.R[b1][b2] += 1
            agent_B.R[b2][b1] += 1
            
            # update V and decide whether to retain or not by the CS function
            ori_cs = agent_B.calculate_CS()
            delta_v = np.random.normal(size=1)
            
            mean = np.mean(agent_B.V)
            if abs(agent_B.V[b1]-mean) <= abs(agent_B.V[b2]-mean):
                weaker = b1
            else:
                weaker = b2
            
            agent_B.V[weaker] += delta_v
            new_cs = agent_B.calculate_CS()
            if new_cs > ori_cs: # CS value increase, keep the change and update the probabilty
                agent_B.update_P()
            else:   # CS value decrease, then don't keep the change
                agent_B.V[weaker] -= delta_v

            # R decays
            agent_B.R *= self.decay_rate


            # record the point for plot img
            if time % self.plot_period == 0:
                print(time)
                self.measurement(time)            
            
            # draw and output the plot map at this time
            if time % 1000 == 0:
                self.draw(time) 

        # output the plot img at the end        
        self.plot_img[0].plot_img(
            self.plot_interpretative_distance[0],self.plot_interpretative_distance[1],time
        )
        self.plot_img[1].plot_img(
            self.plot_mutual_information[0],self.plot_mutual_information[1],time
        )
        self.plot_img[2].plot_img(
            self.plot_Preference_Similarity[0],self.plot_Preference_Similarity[1],time
        )
        self.plot_img[3].plot_img(
            self.plot_Preference_Congruence[0],self.plot_Preference_Congruence[1],time
        )

        
    def draw(self,time):    
        # draw the player's preference of practices in K with plot map
        re_order = []
        now = 0
        re_order.append(self.agents[now].V)
        last = [ i for i in range(1,self.N) ]
        for __ in range(self.N-1):
            most_sim, best_r = 0 ,-2
            for k in last:
                r = np.corrcoef(self.agents[now].V,self.agents[k].V)[0,1]
                if r > best_r :
                    best_r = r
                    most_sim = k
            now = most_sim
            re_order.append(self.agents[now].V)
            last.remove(now)
        self.plotter.plot_map( re_order, time)

    def k_means(self, k, points):
        '''
        the K-means algo for estimate the cluster
        k : assume with k cluster
        points : the points in "K"(size of the set of practices) dimension
        ''' 
        center = []
        t = 100

        # start with k random centers
        for _ in range(k):
            tmp = np.random.uniform(low=-1, high=1, size=(1,self.K))
            center.append(tmp)

        # do t times 
        for _ in range(t):
            cluster = [ list() for i in range(k) ]
            for i in range(self.N):
                best, best_dis = 0, 1
                for j in range(k): # choose the closest
                    dis = (1 - np.corrcoef( points[i] , center[j] )[0,1] )
                    if dis < best_dis :
                        best, best_dis = j, dis
                cluster[best].append( points[i] ) # add the point to the closest cluster

            for j in range(k):  # calculate new center
                if len(cluster[j]) == 0 : continue
                center[j] = np.zeros(self.K)
                for v in cluster[j]:
                    center[j][:] += v[:]
                center[j] /= len(cluster[j])
        
        # finish the K-means algo,
        # then calculate the W_k
        W_k = 0
        for r in range(k):
            if len(cluster[r]) == 0 : continue
            D_r = 0
            for v_i in cluster[r]:
                for v_j in cluster[r]:
                    D_r += (1 - np.corrcoef( v_i , v_j )[0,1])
            W_k += D_r / (2*len(cluster[r]))
        
        return W_k
    
    def cluster_estimate(self,time):
        
        data = [ self.agents[i].V for i in range(self.N) ]
        k_cluster = [0]*(self.K+1)
        for k in range(1,self.K+1):
            k_cluster[k] = np.log(self.k_means(k,data))
        
        ans = 0
        X = 100
        for group in range(100):
            last = -1
            optimal = 100  
            for k in range(1,self.K+1):
                now = self.log_En[k][group] - k_cluster[k]
                if last > now - self.s_k[k] :
                    optimal = k-1
                    break
                last = now
            if optimal == 100: X -= 1
            else : ans += optimal
        ans /= X

        self.plot_cluster_estimate[0].append(time)
        self.plot_cluster_estimate[1].append(ans)

    def expectation_cluster_w(self, num):
        #print("start precalculate expectation_cluster_w")
        self.log_En = [ list() for _ in range(2*self.K+1) ]
        self.s_k = [0]*(self.K+1)
        for _ in range(num):
            print(_)
            reference_data  = [ 
                np.random.uniform(low=-1, high=1, size=(1,self.K))[0] for _ in range(N) 
            ] #reference distribution
            for k in range(1,self.K+1):
                W = self.k_means(k,reference_data)
                self.log_En[k].append( np.log(W) )

        for k in range(1,self.K+1):
            self.s_k[k] = np.std( self.log_En[k] ) / np.sqrt(num)
            print(self.s_k[k])
        #print("Done.")

if __name__ == "__main__":
    img_dir = os.path.join(os.getcwd(), 'imgfiles')
    demo = simulate()
    demo.run()
    demo.plotter.save_gif(fps=100)
    #demo.plotter.save_mp4()