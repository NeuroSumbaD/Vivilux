'''Submodule abstracting live visualizations of hardware data.
'''

import numpy as np
import matplotlib.pyplot as plt

class HeatmapVisualizer:
    def __init__(self, shape=(10, 10), cmap='viridis', vmin=0, vmax=1,
                 xlabel="X", ylabel="Y", clabel="Value", title="Heatmap",
                 origin='upper', scaling=False):
        """
        Initializes the heatmap plot.

        Parameters:
        - shape: Tuple of (rows, cols) for the heatmap.
        - cmap: Colormap for visualization.
        - vmin: Minimum value for color scaling.
        - vmax: Maximum value for color scaling.
        - xlabel: Label for the x-axis.
        - ylabel: Label for the y-axis.
        - clabel: Label for the colorbar.
        - title: Title of the heatmap.
        - origin: 'upper' or 'lower' to set the origin of the heatmap.
        - scaling: If True, dynamically scale the color limits based on data.
        """
        self.shape = shape
        self.data = np.zeros(shape)
        self.vmin = vmin
        self.vmax = vmax
        self.scaling = scaling

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.data, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
        self.colorbar = self.fig.colorbar(self.im, ax=self.ax)

        # Set titles and labels
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.colorbar.set_label(clabel)

        plt.ion()  # Turn on interactive mode
        plt.show()

    def update(self, new_data):
        """
        Update the heatmap with new data.

        Parameters:
        - new_data: A NumPy array of the same shape as initialized.
        """
        if new_data.shape != self.shape:
            raise ValueError(f"Expected shape {self.shape}, got {new_data.shape}")
        self.data = new_data
        self.im.set_data(self.data)
        if self.scaling:
            self.vmin = np.min(new_data)
            self.vmax = np.max(new_data)
            self.im.set_clim(vmin=self.vmin, vmax=self.vmax)  # Optional dynamic scaling
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# Example usage:
if __name__ == "__main__":
    import time
    vis = HeatmapVisualizer(shape=(6, 7), vmin=0, vmax=100,
                            xlabel="Column Index", ylabel="Row Index",
                            clabel="Intensity", origin='upper')

    for i in range(20):
        data = np.random.randint(0, 100, (6, 7))
        vis.update(data)
        time.sleep(0.5)