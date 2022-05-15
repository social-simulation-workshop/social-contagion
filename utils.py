import numpy as  np


class Agent:
    def __init__(self, K):
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
        self.R = np.ones((self.K, self.K))
        self.V = np.random.uniform(low=-1, high=1, size=self.K)
        self.P = np.empty(self.K)
        self.update_P() 


    def update_P(self):
        ''' Use V to update the probability of each practice performed. '''
        eV = np.exp(self.V)
        self.P = eV / np.sum(eV)
    

    def calculate_CS(self):
        ''' Calculate the Constraint Satisfaction (CS). '''
        omega = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(i+1, self.K):
                omega[i][j] = omega[j][i] = abs(self.V[i]-self.V[j])
        omega /= np.max(omega)
        std_R = self.R / np.max(self.R)

        cs = 0
        for i in range(self.K):
            for j in range(self.K):
                cs += abs(std_R[i][j]-omega[i][j])
        
        # Jun: this formula is different from the report.
        cs *= 1/(self.K*(self.K-1))
        return cs


class Simulate:
    def __init__(self, K=6, N=30, decay_rate=0.8, times=100000):
        self.K = K  # a number, the size of the set of the culture practices
        self.N = N # a number, the number of the agents
        self.decay_rate = decay_rate # the decay rate of mattrix R
        self.times = times # the number of iterations
        
        # agents : agent list (init with N agent)
        self.agents = [ Agent(self.K) for _ in range(self.N) ]
        
        # init lists for recording data
        self.rcd_preference_similarity = []
        self.rcd_preference_congruence = []
        self.rcd_interpretative_distance = []
        self.rcd_mutual_information = []
        # self.rcd_cluster_estimate = []
        # self.expectation_cluster_w(100)
    
    
    def get_pref_sim(self):
        return np.array(self.rcd_preference_similarity)
    

    def get_pref_con(self):
        return np.array(self.rcd_preference_congruence)
    

    def get_interp_dis(self):
        return np.array(self.rcd_interpretative_distance)

    
    def get_mul_info(self):
        return np.array(self.rcd_mutual_information)
    

    def record_preference_similarity(self):
        # measure the preference_similarity
        ans = 0
        for i in range(self.N):
            for j in range(i+1,self.N):
                r = np.corrcoef(self.agents[i].V,self.agents[j].V)
                ans += r[0,1]
        ans *= 2/(self.N*(self.N-1))

        self.rcd_preference_similarity.append(ans)


    def record_preference_congruence(self):
        # measure the preference_congruence
        ans = 0
        for i in range(self.N):
            for j in range(i+1,self.N):
                r = np.corrcoef(self.agents[i].V,self.agents[j].V)
                ans += abs(r[0,1])
        ans *= 2/(self.N*(self.N-1))

        self.rcd_preference_congruence.append(ans)


    def record_interpretative_distance(self):
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

        self.rcd_interpretative_distance.append(group_dis)
    

    def record_mutual_information(self):
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

        self.rcd_mutual_information.append(I)
    

    def measure(self):
        ''' Call all the measurement. '''
        self.record_preference_similarity()
        self.record_preference_congruence()
        self.record_interpretative_distance()
        self.record_mutual_information()


    def run(self, log_n=10):
        ''' The main function for simulation. '''
        log_idx, log_t_list = 0, [int(self.times*((i+1)/log_n)) for i in range(log_n)]
        for time in range(self.times):
            if log_t_list[log_idx] == time+1:
                print("t: {:6d}/{:6d} ({:.1f}%) | pref cong: {:.4f}".format(time+1, self.times,
                    100*(time+1)/self.times, self.rcd_preference_congruence[-1]))
                log_idx += 1

            # chose two different agent
            agent_A, agent_B = np.random.choice(self.agents, size=2, replace=False)

            # act and observe
            b1, b2 = b1, b2 = np.random.choice(self.K, size=2, replace=False, p=agent_A.P)
            agent_B.R[b1][b2] += 1
            agent_B.R[b2][b1] += 1
            
            # update V and decide whether to retain or not by the CS function
            ori_cs = agent_B.calculate_CS()
            delta_v = np.random.normal()
            
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
            self.measure()     
            
            # # draw and output the plot map at this time
            # if time % 1000 == 0:
            #     self.draw(time) 

        
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
        self.plotter.plot_map(re_order, time)


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