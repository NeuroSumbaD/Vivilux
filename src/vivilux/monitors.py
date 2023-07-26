import numpy as np
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vivilux import Net, Layer, Mesh

plt.ion()

class Monitor:
    '''A base class for monitoring network activity
    '''
    def __init__(self, name: str,
                 labels: list[str], limits: list[float], numLines: int = 0) -> None:
        self.name = name
        self.xlabel = labels[0]
        self.ylabel = labels[1]
        self.xlim = limits[0]
        self.ylim = limits[1]
     
        #initialize figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        #trace updates
        self.data = np.zeros((self.xlim, numLines))
        self.index = 0
        
        self.lines = self.ax.plot(self.data)

        self.ax.set_title(self.name)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_ylim(0, self.ylim)
        self.ax.legend(range(numLines))


    def update(self, newData: np.array):
        self.data[self.index] = newData

        for lineIndex, line in enumerate(self.lines):
            line.set_ydata(self.data[:, lineIndex])

        self.index = self.index + 1 if self.index < self.xlim-1 else 0

        #update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

