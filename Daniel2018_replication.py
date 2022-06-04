import argparse
import multiprocessing
import numpy as  np

from utils import Simulate
from plot_jun import PlotErrorBarHandler

# param
SMALL_WORLD = True
N_AGENT = 100
N_TIME = 1000000 # N_AGENT * 10000 in the paper
N_TRAIL = 2 # 100 in the paper
LOG_MEASURE_V = N_TIME # log every LOG_MEASURE_V rounds
LOG_VERBOSE_N = 20 # verbose LOG_VERBOSE_N times in total
RNDSEED = 1026

def run_simulation(param_idx, decay_rate, log_data, rnd_seed):
    print("decay_rate = {:.2f} | trail {:3d} started".format(decay_rate, rnd_seed-RNDSEED+1))
    demo = Simulate(N=N_AGENT, decay_rate=decay_rate, times=N_TIME, small_world=SMALL_WORLD,
                    log_measure_v=LOG_MEASURE_V, verbose=False, rnd_seed=rnd_seed)
    demo.run(log_verbose_n=LOG_VERBOSE_N)

    log_data[param_idx].append(demo.get_pref_con(rtn_list=True)[-1])
    print("decay_rate = {:.2f} | trail {:3d} finished | pref cong = {:.4f}".format(decay_rate, rnd_seed-RNDSEED+1, log_data[param_idx][-1]))


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    args = parser.parse_args()
    N_AGENT = args.N
    N_TIME = N_AGENT * 10000

    # multi-processing
    manager = multiprocessing.Manager()
    log_data = [manager.list() for dr in np.arange(1.0, 0.0, -0.1)]
    
    args_list = []
    for param_idx, dr in enumerate(np.arange(1.0, 0.0, -0.1)):
        args_list += [[param_idx, dr, log_data, RNDSEED+trail_idx] for trail_idx in range(N_TRAIL)]
    n_cpus = multiprocessing.cpu_count()
    print("cpu count: {}".format(n_cpus))

    pool = multiprocessing.Pool(n_cpus+2)
    pool.starmap(run_simulation, args_list)

    # result
    log_data_arr = np.array(log_data)
    log_data_mean = np.mean(log_data, axis=-1)
    log_data_std = np.std(log_data, axis=-1)

    # plot
    fn_suffix = "_".join(["N_{}".format(N_AGENT),
                          "t_{}".format(N_TIME),
                          "trail_{}".format(N_TRAIL),
                          "rndseed_{}".format(RNDSEED)])
    plot_handler = PlotErrorBarHandler(xlabel="Decay Rate",
                                       ylabel="Preference Congruence",
                                       title="N={}, {}".format(N_AGENT, "Fully-Connect" if SMALL_WORLD else "Small-World"),
                                       fn="errorbar_{}".format("fully-connected" if SMALL_WORLD else "small-world"),
                                       x_lim=[-0.05, 0.95], y_lim=[-0.05, 1.05], use_ylim=True,
                                       x_tick=[0.0, 0.9, 0.1], y_tick=[0.0, 1.00, 0.25],
                                       figure_size=8, figure_ratio=267/341)
    plot_handler.plot_errorbar(np.arange(0.0, 1.0, 0.1), log_data_mean, log_data_std)
    plot_handler.save_fig(fn_suffix=fn_suffix)
