import multiprocessing
import numpy as  np

# from plot import Plot2DArray
from plot_jun import PlotLinesHandler
from utils import Simulate

# param
N_TIME = 2000 # 100,000 in the paper
N_TRAIL = 5 # 1,000 in the paper
LOG_MEASURE_V = 500 # log every LOG_MEASURE_V rounds
LOG_VERBOSE_N = 20 # verbose LOG_VERBOSE_N times in total
RNDSEED = 1026

def run_simulation(log_data, rnd_seed):
    print("simulation {:4d} started".format(rnd_seed-RNDSEED+1))
    np.random.seed(rnd_seed)
    demo = Simulate(decay_rate=0.95, times=N_TIME, log_measure_v=LOG_MEASURE_V, verbose=False)
    demo.run(log_verbose_n=LOG_VERBOSE_N)

    # log
    pref_con = demo.get_pref_con(rtn_list=True)
    pref_sim = demo.get_pref_sim(rtn_list=True)
    mul_info = demo.get_mul_info(rtn_list=True)
    interp_dis = demo.get_interp_dis(rtn_list=True)
    log_data[0].append(pref_con)
    log_data[1].append(pref_sim)
    log_data[2].append(mul_info)
    log_data[3].append(interp_dis)

    print("simulation {:4d} finished".format(rnd_seed-RNDSEED+1))


def plot_log_data(log_data, use_ylim):
    fn_prefix = "_".join(["multi", 
                          "t_{}".format(N_TIME),
                          "trail_{}".format(N_TRAIL),
                          "rndseed_{}".format(RNDSEED)])
    log_data_arr = np.array(log_data)
    print("log_data_arr size = {}".format(log_data_arr.shape))
    log_data_arr = np.mean(log_data_arr, axis=1)
    print("log_data_arr size = {}".format(log_data_arr.shape))

    # A
    # - Evaluatory Agreement: preference congruence
    plot_handler = PlotLinesHandler(xlabel=None,
                                    ylabel="Preference Congruence",
                                    title="Evaluatory Agreement",
                                    fn="preferenceCongruence",
                                    x_lim=[0, 100000], y_lim=[0.3, 1.0], use_ylim=use_ylim,
                                    x_tick=[25000, 100000, 25000], y_tick=[0.4, 1.0, 0.2],
                                    figure_ratio=7.9/10.0)
    plot_handler.plot_line(log_data_arr[0], data_log_v=LOG_MEASURE_V, linewidth=4, color="#0203ff")
    plot_handler.save_fig(fn_prefix=fn_prefix)

    # - preference similarity
    plot_handler = PlotLinesHandler(xlabel=None,
                                    ylabel=None,
                                    title="Preference Similarity",
                                    fn="preferenceSimilarity",
                                    x_lim=[0, 100000], y_lim=[-4*10**-3, 2*10**-3], use_ylim=use_ylim,
                                    x_tick=[50000, 100000, 50000], y_tick=[-4*10**-3, 2*10**-3, 2*10**-3],
                                    figure_ratio=4.46/6.0)
    plot_handler.plot_line(log_data_arr[1], data_log_v=LOG_MEASURE_V, linewidth=4, color="#0203ff")
    plot_handler.save_fig(fn_prefix=fn_prefix)
    
    # B
    # - Meaningfulness: mutual information
    plot_handler = PlotLinesHandler(xlabel=None,
                                    ylabel="Mutual Information",
                                    title="Meaningfulness",
                                    fn="mutualInformation",
                                    x_lim=[0, 100000], y_lim=[0.2, 1.0], use_ylim=use_ylim,
                                    x_tick=[25000, 100000, 25000], y_tick=[0.2, 1.0, 0.2],
                                    figure_ratio=11.8/15.24)
    plot_handler.plot_line(log_data_arr[2], data_log_v=LOG_MEASURE_V, linewidth=4, color="#0203ff")
    plot_handler.save_fig(fn_prefix=fn_prefix)

    # C
    # - Interpretative Agreement: interpretative distance
    plot_handler = PlotLinesHandler(xlabel=None,
                                    ylabel="Interpretative Distance",
                                    title="Interpretative Agreement",
                                    fn="interpretativeDistance",
                                    x_lim=[0, 100000], y_lim=[0.3, 0.7], use_ylim=use_ylim,
                                    x_tick=[25000, 100000, 25000], y_tick=[0.3, 0.7, 0.1],
                                    figure_ratio=11.8/15.14)
    plot_handler.plot_line(log_data_arr[3], data_log_v=LOG_MEASURE_V, linewidth=4, color="#0203ff")
    plot_handler.save_fig(fn_prefix=fn_prefix)
    


if __name__ == "__main__":
    # multi-processing
    manager = multiprocessing.Manager()
    log_data = [manager.list() for _ in range(4)] ## four measures    
    args_list = [[log_data, RNDSEED+trail_idx] for trail_idx in range(N_TRAIL)]
    n_cpus = multiprocessing.cpu_count()
    print("cpu count: {}".format(n_cpus))

    pool = multiprocessing.Pool(n_cpus+2)
    pool.starmap(run_simulation, args_list)

    # plot
    plot_log_data(log_data, use_ylim=True)
    plot_log_data(log_data, use_ylim=False)
