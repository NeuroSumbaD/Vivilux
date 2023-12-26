from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import imageio

from io import BytesIO
from math import ceil, floor
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from vivilux import Net, Layer, Mesh


class Monitor:
    '''A base class for monitoring network activity
    '''
    def __init__(self, name: str,
                 labels: list[str], limits: list[float],
                 numLines: int = 0, target = "activity") -> None:
        self.name = name
        self.target = target

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

        plt.ion() # Enable interactive plot for real-time updates


    def update(self, newData: dict[str, np.array]):
        self.data[self.index] = newData[self.target]

        for lineIndex, line in enumerate(self.lines):
            line.set_ydata(self.data[:, lineIndex])

        self.index = self.index + 1 if self.index < self.xlim-1 else 0

        #update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class Magnitude(Monitor):
    def __init__(self, name: str, labels: list[str], 
                 limits: list[float], numLines: int = 0,
                 target = "activity") -> None:
        super().__init__(name, labels, limits, numLines, target)
        mag = np.sqrt(np.sum(np.square(self.data), axis=1))
        self.magnitude = self.ax.plot(mag, "--")
        self.ax.legend([*range(numLines), "magnitude"])
    
    def update(self, newData: dict[str, np.array]):
        super().update(newData)
        mag = np.sqrt(np.sum(np.square(self.data), axis=1))
        self.magnitude[0].set_ydata(mag)

        #update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class Multimonitor(Monitor):
    def __init__(self, name: str, labels: list[str], limits: list[float], numLines: int = 0,
                 targets=["activity"], defMonitor = Monitor) -> None:
        self.targets = targets
        self.monitors = []
        for target in targets:
            if target == "gain":
                self.monitors.append(defMonitor(name+f"--({target})", labels,
                                                limits, numLines = 1, target=target))
            else:
                self.monitors.append(defMonitor(name+f"--({target})", labels,
                                                limits, numLines, target=target))
            
        # numMonitor = len(self.monitors)
        # for index, monitor in enumerate(self.monitors):
        #     mgr = monitor.fig.canvas.manager
        #     mgr.window.setGeometry()

    def update(self, newData: dict[str, np.array]):
        for monitor in self.monitors:
            monitor.update(newData)

class Record(Monitor):
    '''A monitor for recording data without plotting.
    '''
    def __init__(self, name: str, labels: list[str], limits: list[float], numLines: int = 0,
                 target="activity") -> None:
        self.name = name
        self.xlabel = labels[0]
        self.ylabel = labels[1]
        self.xlim = limits[0]
        self.ylim = limits[1]

        self.target = target

        #trace updates
        self.data = np.zeros((1,numLines))
    
    def update(self, newData: dict[str, np.array]):
        self.data = np.concatenate((self.data, newData[self.target].reshape(1,-1)))


class Heatmap:
    def __init__(self, net: Net, numEpochs: int, numSamples = 50) -> None:
        self.net = net
        self.numEpochs = numEpochs
        self.numSamples = numSamples
        
        self.validate()

        # Create a directed graph
        self.G = nx.DiGraph()
        
        # Add nodes
        neuronID = []
        for index, layer in enumerate(net.layers):
            length = len(layer)
            for neuron in range(length):
                id = f"L{index}-N{neuron}" if index >0 else f"in-{neuron}"
                neuronID.append(id)
                self.G.add_node(id, subset=index)
                # Define edges
                if index > 0:
                    for snd in range(len(net.layers[index-1])):
                        prevID = f"in-{snd}" if index == 1 else f"L{index-1}-N{snd}"
                        if index == 1:
                            self.G.add_edge(prevID, id, arrowstyle="-|>")
                        else:
                            self.G.add_edge(prevID, id, arrowstyle="<|-|>")
        self.labels = {id: id for id in neuronID}

    def validate(self):
        for layer in self.net.layers:
            if not isinstance(layer.monitor, Record):
                raise TypeError(f"Net must use monitor of type 'Record' (not '{type(layer.monitor)}'). ")

    def draw(self):
        net  = self.net
        labels = self.labels

        # Set up the figure
        plt.figure(figsize=(10.6, 4.4))

        # Draw the graph
        pos = nx.multipartite_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, node_size=1000, node_color='lightblue')
        # Draw edges
        for edge in self.G.edges():
            arrowstyle = self.G.edges[edge]["arrowstyle"]
            nx.draw_networkx_edges(self.G, pos, node_size=1000,
                                   edgelist = [edge],
                                   arrowstyle=arrowstyle,
                                   arrowsize=20,width=2)

        # Add labels to the nodes
        nx.draw_networkx_labels(self.G, pos, labels, font_size=10)

        # Set the title
        plt.title('Neural Network Structure')

        # Show the plot
        plt.show()
    
    def animate(self, fileName, suffix = ".gif"):
        net  = self.net
        labels = self.labels
        numEpochs = self.numEpochs
        numSamples = self.numSamples

        plt.figure(figsize=(10.6, 4.4))
        animation_frames = []

        # Define time and activity axes
        time_steps = len(net.layers[0].monitor.data)
        activity_data = []
        for layer in net.layers:
            activity_data.append(layer.monitor.data)
        activity_data = np.array(activity_data)

        # Create and save animation frames for each time step
        for t in range(time_steps):
            plt.clf()
            ax = plt.gca()
            pos = nx.multipartite_layout(self.G)

            # Draw neurons with activity as node colors
            node_colors = activity_data[:, t, :].flatten()
            nodes = nx.draw_networkx_nodes(
                self.G, pos, node_size=1000, node_color=node_colors,
                cmap=plt.cm.seismic,
                vmax=1.0,
                vmin=0.0
            )
            nodes.set_edgecolor('k')

            
            # Draw edges
            for edge in self.G.edges():
                arrowstyle = self.G.edges[edge]["arrowstyle"]
                nx.draw_networkx_edges(self.G, pos, node_size=1000,
                                    edgelist = [edge],
                                    arrowstyle=arrowstyle,
                                    arrowsize=20,width=2)

            # Add labels to the nodes
            nx.draw_networkx_labels(self.G, pos, labels, font_size=10)

            # Set the title
            title = f'Bidirectional Network (Time Step {t % 50}/50 '
            title += f'[{"Minus" if ceil(t/50)%2 else "Plus"}], '
            title += f'Sample {floor(t/100)%numSamples+1}/{numSamples}, '
            title += f'Epoch {ceil(t/100/numSamples)}/{numEpochs})'
            plt.title(title)
            # Add a color bar legend
            plt.colorbar(nodes, label='Neuron Activation (rate code)')

            
            # Save frame into bytes buffer
            buf = BytesIO()
            plt.savefig(buf, format = "png", dpi = 150)
            buf.seek(0)
            animation_frames.append(buf)

        # Convert frames to a GIF
        output_gif = fileName + suffix
        imageio.mimsave(output_gif,
                        [imageio.imread(frame) for frame in animation_frames],
                        duration=0.1,
                        loop = 0)

        # Clean up buffers
        for buf in animation_frames:
            buf.close()