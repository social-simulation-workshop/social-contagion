import numpy as  np
import os

# from plot import Plot2DArray
from plot_jun import PlotLinesHandler
from utils import TwoAgentsSimulate

RNDSEED = 6
N_TRIALS = 1000
N_TIMES = 1000


if __name__ == "__main__":
    np.random.seed(RNDSEED)

    plot_absCorr = [np.zeros(N_TIMES), np.zeros(N_TIMES)]
    plot_MI = [np.zeros(N_TIMES), np.zeros(N_TIMES)]

    plot_finalCorr_to_initCorr = [[],[]]
    for trail_idx in range(N_TRIALS):
        # simulate for N_TRIALS times
        print("trail {}".format(trail_idx))
        demo = TwoAgentsSimulate(times=N_TIMES)
        final_corr, init_corr =  demo.run()

        plot_finalCorr_to_initCorr[0].append(init_corr)
        plot_finalCorr_to_initCorr[1].append(final_corr)
        plot_absCorr[0] += demo.plot_absCorr[0]
        plot_absCorr[1] += demo.plot_absCorr[1]
        plot_MI[0] += demo.plot_MI[0]
        plot_MI[1] += demo.plot_MI[1]
    
    # get the average of plot_absCorr and plot_MI
    plot_absCorr[0] /= N_TRIALS
    plot_absCorr[1] /= N_TRIALS
    plot_MI[0] /= N_TRIALS
    plot_MI[1] /= N_TRIALS
    
    # plot
    fn_prefix = "_".join(["two", "rndseed_{}".format(RNDSEED)])

    # A
    # - corrlation
    plot_handler = PlotLinesHandler(xlabel="Initial Correlation",
                                    ylabel="Final Correlation",
                                    title="Two-Agent Model",
                                    fn="correlation",
                                    x_lim=[-1, 1], y_lim=[-1, 1],
                                    x_tick=[-1, 1, 0.5], y_tick=[-1, 1, 0.5],
                                    x_as_kilo=False,
                                    figure_size=6.3, figure_ratio=6.58/8.94)
    plot_handler.plot_scatter(plot_finalCorr_to_initCorr[0], plot_finalCorr_to_initCorr[1], color="#1176be", markersize=3)
    plot_handler.save_fig(fn_prefix=fn_prefix)

    # B
    # - absolute correlation and mutual information
    plot_handler = PlotLinesHandler(xlabel="Time",
                                    ylabel=None,
                                    title=None,
                                    fn="absocorr_MI",
                                    x_lim=[0, N_TIMES], y_lim=None,
                                    x_tick=[0, N_TIMES, 200], y_tick=None,
                                    x_as_kilo=False,
                                    figure_size=6.3, figure_ratio=6.58/8.94)
    plot_handler.plot_two_line_two_scale(plot_absCorr[0], plot_absCorr[1], plot_MI[0], plot_MI[1],
                                         y_lim1=[0.2, 1.0], y_tick1=[0.2, 1.0, 0.2],
                                         y_lim2=[0.3, 0.4], y_tick2=[0.3, 0.4, 0.02],
                                         y_label1="Absolute Correlation", y_label2="Mutual Information")
    plot_handler.save_fig(fn_prefix=fn_prefix)
