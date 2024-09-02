from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import imageio

from io import BytesIO
from math import ceil, floor
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from vivilux import Net, Layer, Path


class Monitor:
    '''A base class for monitoring network activity
    '''
    def __init__(self, name: str, labels: list[str], 
                 limits: list[float], numLines: int = 0, 
                 enable = True, legend = True, resizeAxes = False,
                 target = "activity") -> None:
        self.name = name
        self.enable = enable
        self.target = target
        self.resizeAxes = resizeAxes

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
        if legend:
            self.ax.legend(range(numLines))

    def update(self, newData: dict[str, np.array]):
        if self.enable:
            self.data[self.index] = newData[self.target]

            for lineIndex, line in enumerate(self.lines):
                line.set_ydata(self.data[:, lineIndex])
                
            if self.resizeAxes:
                if self.index == 0:
                    self.ymax = 1
                self.ymax = np.max([self.ymax, np.max(self.data)])
                self.ax.set_ylim(0, 1.2*self.ymax)

            self.index = self.index + 1 if self.index < self.xlim-1 else 0

            #update the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

class Magnitude(Monitor):
    def __init__(self, name: str, labels: list[str], 
                 limits: list[float], numLines: int = 0,
                 enable=True, target = "activity",
                 legend = False, resizeAxes = True,
                 **kwargs) -> None:
        super().__init__(name,
                         labels=labels,
                         limits = limits,
                         numLines = numLines,
                         enable = enable, 
                         target=target,
                         legend=legend,
                         resizeAxes=resizeAxes,
                         **kwargs)
        mag = np.sqrt(np.sum(np.square(self.data), axis=1))
        self.magnitude = self.ax.plot(mag, "--")
        if legend:
            self.ax.legend([*range(numLines), "magnitude"])
    
    def update(self, newData: dict[str, np.array]):
        if self.enable:
            super().update(newData)
            mag = np.sqrt(np.sum(np.square(self.data), axis=1))
            self.magnitude[0].set_ydata(mag)
            
            if self.resizeAxes:
                self.ymax = np.max([self.ymax, np.max(mag)])
                self.ax.set_ylim(0, 1.2*self.ymax)

            #update the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

class Multimonitor(Monitor):
    def __init__(self, name: str, labels: list[str], limits: list[float], numLines: int = 0,
                 enable=True, targets=["activity"], defMonitor = Monitor) -> None:
        self.name = name
        self.targets = targets
        self.enable = enable
        self.monitors: list[Monitor] = []
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
            monitor.enable = self.enable
            monitor.update(newData)

class StackedMonitor(Monitor):
    '''A multimonitor that displays multiple monitors in one figure.
    '''
    def __init__(self, name: str, labels: list[str], limits: list[float], 
                 layout: list[int], numLines: int = 0, enable=True, 
                 targets=["activity"], legendVisibility=True) -> None:
        self.name = name
        self.targets = targets
        self.enable = enable
        self.layout = layout

        self.xlabel = labels[0]
        self.ylabel = labels[1]
        self.xlim = limits[0]
        self.ylim = limits[1]
        
        # Generate figure and axes based on layout
        self.sharex = layout[1] == 1
        self.sharey = layout[0] == 1
        self.fig, self.axs = plt.subplots(*self.layout, sharex=self.sharex, sharey=self.sharey)

        # Initialize data for each target
        self.numTarget = len(self.targets)
        self.data = np.zeros((self.numTarget, self.xlim, numLines))
        self.lines = [self.axs[i].plot(self.data[i]) for i in range(self.numTarget)]
        self.index = 0

        # Configure figure and axes labels
        self.fig.suptitle(self.name)
        if self.sharex:
            self.fig.supxlabel(self.xlabel)
        if self.sharey:
            self.fig.supylabel(self.ylabel)

        for i in range(self.numTarget):
            ax = self.axs[i]
            ax.set_title(targets[i])
            if not self.sharex:
                ax.set_xlabel(self.xlabel)
            if not self.sharey:
                ax.set_ylabel(self.ylabel)
            ax.set_ylim(0, self.ylim)
            if legendVisibility:
                ax.legend(range(numLines))

    def update(self, newData: dict[str, np.array]):
        if self.enable:
            for i in range(self.numTarget):
                self.data[i][self.index] = newData[self.targets[i]]

                for lineIndex, line in enumerate(self.lines[i]):
                    line.set_ydata(self.data[i][:, lineIndex])

            self.index = self.index + 1 if self.index < self.xlim-1 else 0

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

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
        self.records = []
        for layer in self.net.layers:
            hasRecord = False
            for monitor in layer.monitors.values():
                if isinstance(monitor, Record):
                    hasRecord = True
                    self.records.append(monitor)
            if not hasRecord:
                raise TypeError(f"Layer [{layer.name}] must use a monitor of type 'Record.'")

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
        cycleLength = np.sum([phase["numTimeSteps"] for phase in net.phaseConfig.values()])

        plt.figure(figsize=(10.6, 4.4))
        animation_frames = []

        # Define time and activity axes
        time_steps = len(self.records[0].data)
        activity_data = []
        for record in self.records:
            activity_data.append(record.data)
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
            # Calculate time step within the current cycle
            cycleStep = t % cycleLength
            for phase in net.phaseConfig:
                if cycleStep <= net.phaseConfig[phase]["numTimeSteps"]:
                    currentPhase = phase
                    phaseLength = net.phaseConfig[phase]["numTimeSteps"]
                    break
                else:
                    cycleStep -= net.phaseConfig[phase]["numTimeSteps"]
            title = f'Bidirectional Network (Time Step {cycleStep}/{phaseLength} '
            title += f'[{currentPhase}], '
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