import argparse
from matplotlib.pyplot import plot
import numpy as  np
import random
from main import simulate
from plot import Plot2DArray
import os
# init
np.random.seed(34)

class two_agents_simulate(simulate):
    
    def __init__(self):    
        super().__init__( K=6, N=2, decay_rate = 0.9 )
        self.init_corr = np.corrcoef(self.agents[0].V,self.agents[1].V) # 圖A的橫軸
        self.plotter2 = Plot2DArray(filename_prefix = "absCorr and MI")
        self.plot_absCorr = [[],[]]
        self.plot_MI = [[],[]]
        
    def MI(self):
        ##### mutual information
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
            agent_B.R *= self.decay_rate

            if time %1000 == 0:
                self.draw(time)
                # 圖b 橫軸是時間  縱軸分兩條線 
                # 左邊座標軸 Absolute Correlation
                # 右邊座標軸 Mutual Information -> 跟上面的 mutual information 的東西一樣
                I = self.MI()   # 我把他自己弄了個func MI
                abs_corr = abs(np.corrcoef(self.agents[0].V,self.agents[1].V)[0,1])
                self.plot_MI[0].append(time)
                self.plot_MI[1].append(I)
                self.plot_absCorr[0].append(time)
                self.plot_absCorr[1].append(abs_corr)
        ################
        ############### 圖A 是等你跑完後 用 final_corr 對 init_corr 做圖
        ################ 他說跑1000次 應該是1000個點的意思
        final_corr = np.corrcoef(self.agents[0].V,self.agents[1].V)[0,1] # 圖A的縱軸
        return (final_corr,self.init_corr)

if __name__ == "__main__":
    img_dir = os.path.join(os.getcwd(), 'imgfiles')
    
    plot_finalCorr_to_initCorr = [[],[]]
    for time in range(1000):
        demo = two_agents_simulate()
        final_corr, init_corr =  demo.run()
        plot_finalCorr_to_initCorr[0].append(init_corr)
        plot_finalCorr_to_initCorr[1].append(final_corr)
        if time == 0 :
            demo.plotter2.plot_2line_img(
                demo.plot_absCorr[0],
                demo.plot_absCorr[1],
                demo.plot_MI[0],
                demo.plot_MI[1],
                time
            )
            demo.plotter.save_gif(fps=100)
            demo.plotter.save_mp4()
    plotter = Plot2DArray(filename_prefix = "finalCorr_to_initCorr") # visual
    plotter.plot_img(plot_finalCorr_to_initCorr[0], plot_finalCorr_to_initCorr[1], 1000)
    
    