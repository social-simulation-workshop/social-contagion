import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

class Plot2DArray:
    def __init__(self, filename_prefix="", output_dir="imgfiles"):
        super().__init__()

        # use the current time as filename if not specified
        if filename_prefix:
            self.filename_prefix = filename_prefix
        else:
            self.filename_prefix = "simulation_{}".format(datetime.datetime.now().strftime('%m_%d_%H_%M'))
        self.output_dir=output_dir

        self.max_digit = 4
        self.plotted_img_paths = []
    def plot_map(self, map, t, cmap="magma", figure_size=(9, 9)):
        """
        Param
        - map: np.array
            an 2d numpy array to plot
        - t: float
            current timestep t
        - cmap:
            the color set for meshcolor.
            you can choose the one you like at https://matplotlib.org/stable/tutorials/colors/colormaps.html
        """
        title = "t = {:.3f}".format(t)
        output_path = os.path.join(os.getcwd(), self.output_dir, self.filename_prefix)
        filename = "{}_{:.3f}.png".format(self.filename_prefix, t)
        
        plt.figure(figsize=figure_size, dpi=80)
        plt.title(title)
        plt.imshow(map, cmap=cmap, aspect="auto")
        plt.colorbar()
        self.plotted_img_paths.append(self._save_fig(output_path, filename, t))
        plt.close()

    def _save_fig(self, output_path, fn, t):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_path = os.path.join(output_path, fn)
        plt.savefig(file_path)
        print("|t={}| figrue saved to {}".format((str(t)+' '*self.max_digit)[:self.max_digit], file_path))
        return file_path

    def save_gif(self, fps=30, img_dir=""):
        filename = "{}.gif".format(self.filename_prefix)
        file_path = os.path.join(os.getcwd(), self.output_dir, filename)
        
        # img paths
        all_img_paths = self.plotted_img_paths
        if img_dir:
            all_img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

        images = [imageio.imread(img_path) for img_path in all_img_paths]
        imageio.mimsave(file_path, images, duration=1/fps)
        print("gif saved to {}".format(file_path))

    
    def save_mp4(self, fps=30, img_dir=""):
        filename = "{}.mp4".format(self.filename_prefix)
        file_path = os.path.join(os.getcwd(), self.output_dir, filename)

        # img paths
        all_img_paths = self.plotted_img_paths
        if img_dir:
            all_img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        
        writer = imageio.get_writer(file_path, fps=20)
        for img_path in all_img_paths:
            writer.append_data(imageio.imread(img_path))
        writer.close()
        print("mp4 saved to {}".format(file_path))



if __name__ == "__main__":
    img_dir = os.path.join(os.getcwd(), 'imgfiles')
    #plotter = Plot2DArray()
    #plotter.save_gif(img_dir=img_dir)
    #plotter.save_mp4(img_dir=img_dir)


    # usage example
    t = 60
    lots_of_data = np.random.randint(256, size=(t, 128, 6))
    plotter = Plot2DArray()
    for i in range(t):
        plotter.plot_map(lots_of_data[i], i)
    plotter.save_gif()
    plotter.save_mp4()