import numpy as  np
import itertools

class Agent:
    _ids = itertools.count(0)

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
        self.id = next(self._ids)
        self.K = K
        self.R = np.ones((self.K, self.K))
        self.V = np.random.uniform(low=-1, high=1, size=self.K)
        self.P = np.empty(self.K)
        self.update_P()
    

    def __eq__(self, other) -> bool:
        if not isinstance(other, Agent):
            raise NotImplementedError("Agent compared to non-Agent obejct.")
        return self.id == other.id
    

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
        
        cs *= self.K / (self.K*(self.K-1))
        return cs


class Simulate:
    def __init__(self, K=6, N=30, decay_rate=0.8, times=100000,
                 log_measure_v=1, verbose=True, rnd_seed=np.random.randint(10000)):
        np.random.seed(rnd_seed)
        self.rnd_seed = rnd_seed

        self.K = K  # a number, the size of the set of the culture practices
        self.N = N # a number, the number of the agents
        self.decay_rate = decay_rate # the decay rate of mattrix R
        self.times = times # the number of iterations
        self.log_measure_v = log_measure_v
        self.verbose = verbose
        
        # agents : agent list (init with N agent)
        self.agents = [ Agent(self.K) for _ in range(self.N) ]
        
        # init lists for recording data
        self.rcd_preference_similarity = []
        self.rcd_preference_congruence = []
        self.rcd_interpretative_distance = []
        self.rcd_mutual_information = []
        # self.rcd_cluster_estimate = []
        # self.expectation_cluster_w(100)
    
    
    def get_pref_sim(self, rtn_list=False):
        if rtn_list:
            return self.rcd_preference_similarity
        else:
            return np.array(self.rcd_preference_similarity)
    

    def get_pref_con(self, rtn_list=False):
        if rtn_list:
            return self.rcd_preference_congruence
        else:
            return np.array(self.rcd_preference_congruence)
    

    def get_interp_dis(self, rtn_list=False):
        if rtn_list:
            return self.rcd_interpretative_distance
        else:
            return np.array(self.rcd_interpretative_distance)

    
    def get_mul_info(self, rtn_list=False):
        if rtn_list:
            return self.rcd_mutual_information
        else:
            return np.array(self.rcd_mutual_information)
    

    def record_preference_similarity_congruence(self):
        # measure the preference_similarity
        sim, con = 0, 0
        
        for i in range(self.N):
            for j in range(i+1,self.N):
                r = np.corrcoef(self.agents[i].V, self.agents[j].V)
                sim += r[0, 1]
                con += abs(r[0, 1])
        sim *= 2/(self.N*(self.N-1))
        con *= 2/(self.N*(self.N-1))

        self.rcd_preference_similarity.append(sim)
        self.rcd_preference_congruence.append(con)


    def record_interpretative_distance(self):
        """ Measure the interpretative_distance at the group level. """
        group_dis = 0
        count = 0
        for ag_i in self.agents:
            for ag_j in self.agents:
                if ag_i == ag_j:
                    continue
                count += 1
                ag_i_norm_R = ag_i.R / np.max(ag_i.R) if np.max(ag_i.R) != 0 else ag_i.R
                ag_j_norm_R = ag_j.R / np.max(ag_j.R) if np.max(ag_j.R) != 0 else ag_j.R
                group_dis += np.mean(np.abs(ag_i_norm_R-ag_j_norm_R))
        print(count, group_dis/count)
        group_dis /= (self.N**2)

        self.rcd_interpretative_distance.append(group_dis)
    

    def record_mutual_information(self):
        """ Measure the mutual information. """
        # Grace's implementation at the workshop.
        # I = 0
        # for x in range(self.K):
        #     p_x = 0 # the variable for P(b1 = x)
        #     for agent in self.agents:
        #         p_x += agent.P[x]
        #     p_x /= self.N

        #     for y in range(self.K):
        #         if y == x:  # P(b1 = x, b2 = x) = 0 given condition
        #             continue
        #         p_y = 0 # the variable for P(b2 = y)
        #         p_x_y = 0 # the variable for P(b1 = x, b2 = y)
        #         for agent in self.agents:
        #             # enumerate X(b1) (using variable j) to get the marginal probability of y
        #             for j in range(self.K): 
        #                 if j == y: continue
        #                 p_y +=  agent.P[j] * agent.P[y] / (1-agent.P[j])
        #             p_x_y += agent.P[x]*agent.P[y] / (1-agent.P[x])
        #         p_y /= self.N
        #         p_x_y /= self.N
        #         I += p_x_y*np.log(p_x_y/p_x/p_y)
        
        # Jun's implementation
        # see Appendix: Measurement
        p_ag_x_y = np.empty((self.N, self.K, self.K))
        for ag_idx in range(self.N):
            for x in range(self.K):
                for y in range(self.K):
                    p_ag = self.agents[ag_idx].P
                    if x == y or p_ag[x] == 1.0:
                        p_ag_x_y[ag_idx][x][y] = 0
                    else:
                        p_ag_x_y[ag_idx][x][y] = p_ag[x] * p_ag[y] / (1-p_ag[x])
        
        p_ag_y = np.sum(p_ag_x_y, axis=1)
        p_y = np.mean(p_ag_y, axis=0)
        p_x_y = np.mean(p_ag_x_y, axis=0)
        p_x = np.mean(np.array([ag.P for ag in self.agents]), axis=0)

        # method 1
        # I = 0
        # I_x_y = np.zeros((self.K, self.K))
        # for x in range(self.K):
        #     for y in range(self.K):
        #         if x == y:
        #             continue
        #         tmp = p_x_y[x][y] * np.log(p_x_y[x][y]/(p_x[x]*p_y[y]))
        #         I_x_y[x][y] = tmp
        #         if np.isnan(tmp):
        #             raise ValueError
        #         I += tmp
        
        # method 2
        p_y_2d, p_x_2d = np.meshgrid(p_y, p_x)
        I_x_y = np.multiply(p_x_y, np.log2(np.divide(p_x_y, np.multiply(p_x_2d, p_y_2d)))) # /0 warning already handled
        np.fill_diagonal(I_x_y, 0)
        I = np.sum(I_x_y)

        self.rcd_mutual_information.append(I)
    

    def measure(self):
        ''' Call all the measurement. '''
        self.record_preference_similarity_congruence()
        self.record_interpretative_distance()
        self.record_mutual_information()


    def run(self, log_verbose_n=10):
        ''' The main function for simulation. '''
        print("params: K={}, N={}, decay_rate={}, times={}".format(self.K, self.N, self.decay_rate, self.times))
        print("opt args: rnd_seed={}, log_measure_v={}, verbose={}, log_verbose_n={}".format(self.rnd_seed,
            self.log_measure_v, self.verbose, log_verbose_n))
        log_idx, log_t_list = 0, [int(self.times*((i+1)/log_verbose_n)) for i in range(log_verbose_n)]
        for time in range(self.times):
            if log_t_list[log_idx] == time+1:
                if self.verbose and self.rcd_preference_congruence:
                    print("t: {:6d}/{:6d} ({:.1f}%) | pref cong: {:.4f}".format(time+1, self.times,
                        100*(time+1)/self.times, self.rcd_preference_congruence[-1]))
                log_idx += 1

            # chose two different agent
            agent_A, agent_B = np.random.choice(self.agents, size=2, replace=False)

            # act and observe
            b1, b2 = np.random.choice(self.K, size=2, replace=False, p=agent_A.P)
            agent_B.R[b1][b2] += 1
            agent_B.R[b2][b1] += 1
            assert b1 != b2
            
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
            # agent_B.R[b1][b2] *= self.decay_rate
            # agent_B.R[b2][b1] *= self.decay_rate
            agent_B.R *= self.decay_rate


            # record the point for plot img
            if (time+1) % self.log_measure_v == 0:
                self.measure()     
            
            # # draw and output the plot map at this time
            # if time % 1000 == 0:
            #     self.draw(time) 

        
    # def draw(self,time):    
    #     # draw the player's preference of practices in K with plot map
    #     re_order = []
    #     now = 0
    #     re_order.append(self.agents[now].V)
    #     last = [ i for i in range(1,self.N) ]
    #     for __ in range(self.N-1):
    #         most_sim, best_r = 0 ,-2
    #         for k in last:
    #             r = np.corrcoef(self.agents[now].V,self.agents[k].V)[0,1]
    #             if r > best_r :
    #                 best_r = r
    #                 most_sim = k
    #         now = most_sim
    #         re_order.append(self.agents[now].V)
    #         last.remove(now)
    #     self.plotter.plot_map(re_order, time)


    # def k_means(self, k, points):
    #     '''
    #     the K-means algo for estimate the cluster
    #     k : assume with k cluster
    #     points : the points in "K"(size of the set of practices) dimension
    #     ''' 
    #     center = []
    #     t = 100

    #     # start with k random centers
    #     for _ in range(k):
    #         tmp = np.random.uniform(low=-1, high=1, size=(1,self.K))
    #         center.append(tmp)

    #     # do t times 
    #     for _ in range(t):
    #         cluster = [ list() for i in range(k) ]
    #         for i in range(self.N):
    #             best, best_dis = 0, 1
    #             for j in range(k): # choose the closest
    #                 dis = (1 - np.corrcoef( points[i] , center[j] )[0,1] )
    #                 if dis < best_dis :
    #                     best, best_dis = j, dis
    #             cluster[best].append( points[i] ) # add the point to the closest cluster

    #         for j in range(k):  # calculate new center
    #             if len(cluster[j]) == 0 : continue
    #             center[j] = np.zeros(self.K)
    #             for v in cluster[j]:
    #                 center[j][:] += v[:]
    #             center[j] /= len(cluster[j])
        
    #     # finish the K-means algo,
    #     # then calculate the W_k
    #     W_k = 0
    #     for r in range(k):
    #         if len(cluster[r]) == 0 : continue
    #         D_r = 0
    #         for v_i in cluster[r]:
    #             for v_j in cluster[r]:
    #                 D_r += (1 - np.corrcoef( v_i , v_j )[0,1])
    #         W_k += D_r / (2*len(cluster[r]))
        
    #     return W_k
    

    # def cluster_estimate(self,time):
    #     data = [ self.agents[i].V for i in range(self.N) ]
    #     k_cluster = [0]*(self.K+1)
    #     for k in range(1,self.K+1):
    #         k_cluster[k] = np.log(self.k_means(k,data))
        
    #     ans = 0
    #     X = 100
    #     for group in range(100):
    #         last = -1
    #         optimal = 100  
    #         for k in range(1,self.K+1):
    #             now = self.log_En[k][group] - k_cluster[k]
    #             if last > now - self.s_k[k] :
    #                 optimal = k-1
    #                 break
    #             last = now
    #         if optimal == 100: X -= 1
    #         else : ans += optimal
    #     ans /= X

    #     self.plot_cluster_estimate[0].append(time)
    #     self.plot_cluster_estimate[1].append(ans)


    # def expectation_cluster_w(self, num):
    #     #print("start precalculate expectation_cluster_w")
    #     self.log_En = [ list() for _ in range(2*self.K+1) ]
    #     self.s_k = [0]*(self.K+1)
    #     for _ in range(num):
    #         print(_)
    #         reference_data  = [ 
    #             np.random.uniform(low=-1, high=1, size=(1,self.K))[0] for _ in range(N) 
    #         ] #reference distribution
    #         for k in range(1,self.K+1):
    #             W = self.k_means(k,reference_data)
    #             self.log_En[k].append( np.log(W) )

    #     for k in range(1,self.K+1):
    #         self.s_k[k] = np.std( self.log_En[k] ) / np.sqrt(num)
    #         print(self.s_k[k])
    #     #print("Done.")


class TwoAgentsSimulate(Simulate):
    
    def __init__(self, times=1000):    
        super().__init__(K=6, N=2, decay_rate=0.8, times=times)
        # correlation at the begin
        self.init_corr = np.corrcoef(self.agents[0].V,self.agents[1].V)[0,1]
        self.plot_absCorr = [[],[]]
        self.plot_MI = [[],[]]
    

    def MI(self):
        # calculate mutual information
        I = 0   # mutual information
        for x in range(self.K):
            p_x = 0 # P(b1 = x)
            for agent in self.agents:
                p_x += agent.P[x]
            p_x /= self.N
            for y in range(self.K):
                if y == x:  # P(b1 = x, b2 = x) = 0 given condition
                    continue
                p_y = 0 # P(b2 = y)
                p_x_y = 0 # P(b1 = x, b2 = y)
                for agent in self.agents:
                    for j in range(self.K):  # enumerate X(b1) (using variable j) to get the marginal probability of y
                        if j == y: continue
                        p_y +=  agent.P[j] * agent.P[y]/( 1 - agent.P[j])
                    p_x_y += agent.P[x]*agent.P[y]/(1-agent.P[x])
                p_y /= self.N
                p_x_y /= self.N
                I += p_x_y*np.log2(p_x_y/p_x/p_y)
        return I
    

    def run(self):
        for time in range(self.times):
            # the two people change their roles every time
            agent_A = self.agents[time % 2]
            agent_B = self.agents[(time+1) % 2]

            # act and observe
            b1, b2 = np.random.choice(self.K, size=2, replace=False, p=agent_A.P)
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
            agent_B.R *= self.decay_rate

            # calculate and record the measurement
            I = self.MI()
            # Absolute Correlation at this time
            abs_corr = abs(np.corrcoef(self.agents[0].V,self.agents[1].V)[0,1])
            self.plot_MI[0].append(time)
            self.plot_MI[1].append(I)
            self.plot_absCorr[0].append(time)
            self.plot_absCorr[1].append(abs_corr)

        # correlation at the end
        final_corr = np.corrcoef(self.agents[0].V,self.agents[1].V)[0,1]
        return (final_corr,self.init_corr)


if __name__ == "__main__":
    demo = Simulate(decay_rate=0.95)
    demo.record_interpretative_distance()
