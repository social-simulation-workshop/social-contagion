import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import numpy as np
import os
import seaborn as sns

sns.set()
sns.set_style("whitegrid")

class PlotLinesHandler:
    _ids = itertools.count(0)
    EPSILON = 10**-5

    def __init__(self, title, xlabel, ylabel, fn,
        x_lim, y_lim, x_tick, y_tick, figure_ratio, figure_size=5,
        output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgfiles")) -> None:
        
        self.id = next(self._ids)

        self.output_dir = output_dir
        self.fn = fn
        self.legend_list = list()

        plt.figure(self.id, figsize=(figure_size, figure_size*figure_ratio), dpi=160)
        plt.title(title, fontweight="bold")
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)

        ax = plt.gca()
        if x_lim is not None:
            ax.set_xlim([x_lim[0], x_lim[1]])
            x_tick_label = ["{}K".format(int(i/1000)) for i in np.arange(x_tick[0], x_tick[1]+self.EPSILON, step=x_tick[2])]
            plt.xticks(np.arange(x_tick[0], x_tick[1]+self.EPSILON, step=x_tick[2]), x_tick_label)
            # ax.xaxis.set_major_locator(LinearLocator(int(x_lim/20)+1))
            # ax.xaxis.set_major_formatter('{x:0.0f}')
        if y_lim is not None:
            ax.set_ylim([y_lim[0], y_lim[1]])
            plt.yticks(np.arange(y_tick[0], y_tick[1]+self.EPSILON, step=y_tick[2]))
            # ax.yaxis.set_major_locator(LinearLocator(int(y_lim/10)+1))
            # ax.yaxis.set_major_formatter('{x:0.0f}')

    def plot_line(self, data, linewidth=1, color="", alpha=1.0):
        plt.figure(self.id)
        if color:
            plt.plot(np.arange(data.shape[-1]), data*100,
                linewidth=linewidth, color=color, alpha=alpha)
        else:
            plt.plot(np.arange(data.shape[-1]), data*100, linewidth=linewidth)
    
    def plot_changes(self, inno_id_list, data, line_width=1, color=""):
        plt.figure(self.id)

        last_inno = inno_id_list[0]
        for step in range(1, len(inno_id_list)):
            if inno_id_list[step] != last_inno:
                if color:
                    plt.axvline(x=step, ymin=0, ymax=1, linewidth=line_width, color=color)
                else:
                    plt.axvline(x=step, ymin=0, ymax=1, linewidth=line_width)
                ax = plt.gca()
                ax.text(step+1, data[step]*100-5, "inv "+str(inno_id_list[step]),
                        ha="center", va="center", fontsize=10)
                last_inno = inno_id_list[step]

    def save_fig(self, fn_prefix="", fn_suffix=""):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        plt.figure(self.id)
        fn = "_".join([fn_prefix, self.fn, fn_suffix]) + ".png"
        
        # plt.legend([title_param.split("_")[-1]])
        # plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        plt.savefig(os.path.join(self.output_dir, fn))
        print("fig save to {}".format(os.path.join(self.output_dir, fn)))
