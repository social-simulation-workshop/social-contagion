import numpy as  np
import os

# from plot import Plot2DArray
from plot_jun import PlotLinesHandler
from utils import Simulate


RNDSEED = 6

if __name__ == "__main__":
    np.random.seed(RNDSEED)

    # simulate
    t = 20
    demo = Simulate(times=t)
    demo.run()

    # plot
    fn_prefix = "_".join(["multi", "t_{}".format(t), "rndseed_{}".format(RNDSEED)])

    # A
    # - Evaluatory Agreement: preference congruence
    plot_handler = PlotLinesHandler(xlabel=None,
                                    ylabel="Preference Congruence",
                                    title="Evaluatory Agreement",
                                    fn="preferenceCongruence",
                                    x_lim=[0, 100000], y_lim=[0.3, 1.0],
                                    x_tick=[25000, 100000, 25000], y_tick=[0.4, 1.0, 0.2],
                                    figure_ratio=7.9/10.0)
    pref_con = demo.get_pref_con()
    plot_handler.plot_line(pref_con, linewidth=4, color="#0203ff")
    plot_handler.save_fig(fn_prefix=fn_prefix)

    # - preference similarity
    plot_handler = PlotLinesHandler(xlabel=None,
                                    ylabel=None,
                                    title="Preference Similarity",
                                    fn="preferenceSimilarity",
                                    x_lim=[0, 100000], y_lim=[-4*10**-3, 2*10**-3],
                                    x_tick=[50000, 100000, 50000], y_tick=[-4*10**-3, 2*10**-3, 2*10**-3],
                                    figure_ratio=4.46/6.0)
    pref_sim = demo.get_pref_sim()
    plot_handler.plot_line(pref_sim, linewidth=4, color="#0203ff")
    plot_handler.save_fig(fn_prefix=fn_prefix)
    
    # B
    # - Meaningfulness: mutual information
    plot_handler = PlotLinesHandler(xlabel=None,
                                    ylabel="Mutual Information",
                                    title="Meaningfulness",
                                    fn="mutualInformation",
                                    x_lim=[0, 100000], y_lim=[0.2, 1.0],
                                    x_tick=[25000, 100000, 25000], y_tick=[0.2, 1.0, 0.2],
                                    figure_ratio=11.8/15.24)
    mul_info = demo.get_mul_info()
    plot_handler.plot_line(mul_info, linewidth=4, color="#0203ff")
    plot_handler.save_fig(fn_prefix=fn_prefix)

    # C
    # - Interpretative Agreement: interpretative distance
    plot_handler = PlotLinesHandler(xlabel=None,
                                    ylabel="Interpretative Distance",
                                    title="Interpretative Agreement",
                                    fn="interpretativeDistance",
                                    x_lim=[0, 100000], y_lim=[0.3, 0.7],
                                    x_tick=[25000, 100000, 25000], y_tick=[0.3, 0.7, 0.1],
                                    figure_ratio=11.8/15.14)
    interp_dis = demo.get_interp_dis()
    plot_handler.plot_line(interp_dis, linewidth=4, color="#0203ff")
    plot_handler.save_fig(fn_prefix=fn_prefix)
