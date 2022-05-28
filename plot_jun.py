import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

sns.set()
# sns.set_style("white")
sns.set_style("whitegrid")


class PlotLinesHandler:
    _ids = itertools.count(0)
    EPSILON = 10**-5

    def __init__(self, title, xlabel, ylabel, fn,
        x_lim, y_lim, x_tick, y_tick, figure_ratio, use_ylim=True, figure_size=8, x_as_kilo=True,
        output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgfiles")) -> None:
        
        self.id = next(self._ids)

        self.output_dir = output_dir
        self.fn = fn
        self.legend_list = list()

        plt.figure(self.id, figsize=(figure_size, figure_size*figure_ratio), dpi=160)
        if title is not None:
            plt.title(title, fontweight="bold")
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)

        ax = plt.gca()
        if x_lim is not None:
            ax.set_xlim([x_lim[0], x_lim[1]])
            if x_as_kilo:
                x_tick_label = ["{}K".format(int(i/1000)) for i in np.arange(x_tick[0], x_tick[1]+self.EPSILON, step=x_tick[2])]
                plt.xticks(np.arange(x_tick[0], x_tick[1]+self.EPSILON, step=x_tick[2]), x_tick_label)
            else:
                plt.xticks(np.arange(x_tick[0], x_tick[1]+self.EPSILON, step=x_tick[2]))
        if y_lim is not None and use_ylim:
            ax.set_ylim([y_lim[0], y_lim[1]])
            plt.yticks(np.arange(y_tick[0], y_tick[1]+self.EPSILON, step=y_tick[2]))
        self.use_ylim = use_ylim
            

    def plot_line(self, data, data_log_v=1, linewidth=1, color="", alpha=1.0):
        plt.figure(self.id)
        if color:
            plt.plot((np.arange(data.shape[-1])+1)*data_log_v, data,
                linewidth=linewidth, color=color, alpha=alpha)
        else:
            plt.plot((np.arange(data.shape[-1])+1)*data_log_v, data,
                linewidth=linewidth)
    
    def plot_two_line_two_scale(self, x1, y1, x2, y2, 
                                y_lim1, y_tick1, y_lim2, y_tick2,
                                y_label1, y_label2,
                                color1="#000000", color2="#797979"):
        ax1 = plt.gca()
        ax1.plot(x1, y1, color=color1)
        ax1.set_ylabel(y_label1, color=color1)
        ax1.set_ylim([y_lim1[0], y_lim1[1]])
        ax1.set_yticks(np.arange(y_tick1[0], y_tick1[1]+self.EPSILON, step=y_tick1[2]))
        ax1.tick_params(axis='y', colors=color1)

        ax2 = ax1.twinx()
        ax2.plot(x2, y2, color=color2)
        ax2.set_ylabel(y_label2, color=color2)
        ax2.set_ylim([y_lim2[0], y_lim2[1]])
        ax2.set_yticks(np.arange(y_tick2[0], y_tick2[1]+self.EPSILON, step=y_tick2[2]))
        ax2.tick_params(axis='y', colors=color2)


    def plot_scatter(self, x, y, color="", markersize=1):
        plt.figure(self.id)
        if color:
            plt.scatter(x, y, s=markersize, color=color)
        else:
            plt.scatter(x, y, s=markersize)

    def save_fig(self, fn_prefix="", fn_suffix=""):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        plt.figure(self.id)
        if not self.use_ylim:
            fn_suffix += "y_unlimited"
        fn = "_".join([fn_prefix, self.fn, fn_suffix]) + ".png"
        
        # plt.legend([title_param.split("_")[-1]])
        # plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        plt.savefig(os.path.join(self.output_dir, fn))
        print("fig save to {}".format(os.path.join(self.output_dir, fn)))
