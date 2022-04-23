# social-contagion
This is the implementation of paper no.16 in 中研院社會模擬工作坊 2021.

[Beyond Social Contagion: Associative Diffusion and the Emergence of Cultural Variation (Goldberg, A., & Stein, S. K., 2018)](https://journals.sagepub.com/doi/pdf/10.1177/0003122418797576)

Team 5: 擬會作社模

## introduction
主要有三份 python 檔案，分別為 main.py, two_agent.py, plot.py(畫出結果的檔案)。

two-agent: 在 python3 執行 two_agent.py 就可以了（也可以透過改物件 two_agents_simulate() 的參數，來嘗試不一樣的參數底下的模擬）

multi-agent:  在 python3 執行 main.py 就可以了 （也可以透過改物件 simulate() 的參數，來嘗試不一樣的參數底下的模擬）。

* 因為 two_agents_simulate() 在物件上繼承了 simulate() ，若更動了 simulate() 也有可能影響 two_agents_simulate()
* 可以透過 np.random.seed 來控制亂數產生的結果。（如果要不能預測的結果，也可以把他註解掉）
