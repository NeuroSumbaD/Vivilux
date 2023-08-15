from copy import deepcopy
import os
from io import BytesIO
from math import ceil, floor

from vivilux import *
import vivilux as vl
from vivilux.learningRules import CHL, GeneRec
from vivilux.optimizers import Momentum

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import networkx as nx
import imageio
np.random.seed(seed=0)


numEpochs = 5
numSamples = 5


#define input and output data (must be normalized and positive-valued)
iris = datasets.load_iris()
inputs = iris.data
maxMagnitude = np.max(np.sqrt(np.sum(np.square(inputs), axis=1)))
inputs = inputs/maxMagnitude # bound on (0,1]
targets = np.zeros((len(inputs),4))
targets[np.arange(len(inputs)), iris.target] = 1
#shuffle both arrays in the same manner
shuffle = np.random.permutation(len(inputs))
inputs, targets = inputs[shuffle][:numSamples], targets[shuffle][:numSamples]


netMixed = RecurNet([
    Layer(4, isInput=True),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=CHL)
    ],
    Mesh, learningRate = 0.01,
    monitoring=True, defMonitor = vl.monitors.Record,
    name = "NET_Mixed")

# Create a directed graph
G = nx.DiGraph()

neuronID = []
# Add nodes
for index, layer in enumerate(netMixed.layers):
    length = len(layer)
    for neuron in range(length):
        id = f"L{index}-N{neuron}" if index >0 else f"in-{neuron}"
        neuronID.append(id)
        G.add_node(id, subset=index)

        if index > 0:
            for snd in range(len(netMixed.layers[index-1])):
                prevID = f"in-{snd}" if index == 1 else f"L{index-1}-N{snd}"
                if index == 1:
                    G.add_edge(prevID, id, arrowstyle="-|>")
                else:
                    G.add_edge(prevID, id, arrowstyle="<|-|>")



resultMixed = netMixed.Learn(inputs, targets,
                             numEpochs=numEpochs,
                             reset=False)



# Set up the figure
# plt.figure(figsize=(10.6, 4.4))

# # Draw the graph
# pos = nx.multipartite_layout(G)
# nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
# # Draw edges
# for edge in G.edges():
#     arrowstyle = G.edges[edge]["arrowstyle"]
#     nx.draw_networkx_edges(G, pos, node_size=1000,
#                            edgelist = [edge],
#                            arrowstyle=arrowstyle,
#                            arrowsize=20,width=2)

# # Add labels to the nodes
labels = {id: id for id in neuronID}
# nx.draw_networkx_labels(G, pos, labels, font_size=10)

# # Set the title
# plt.title('Neural Network Structure')

# # Show the plot
# plt.show()

plt.ioff()
output_folder = 'HeatmapFrames'
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(10.6, 4.4))
animation_frames = []

# Define time and activity axes
time_steps = len(netMixed.layers[0].monitor.data)
activity_data = []
for layer in netMixed.layers:
    activity_data.append(layer.monitor.data)
activity_data = np.array(activity_data)

# Create and save animation frames for each time step
for t in range(time_steps):
    plt.clf()
    ax = plt.gca()
    pos = nx.multipartite_layout(G)

    # Draw neurons with activity as node colors
    node_colors = activity_data[:, t, :].flatten()
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=1000, node_color=node_colors,
        cmap=plt.cm.seismic,
        vmax=1.0,
        vmin=0.0
    )
    nodes.set_edgecolor('k')

    
    # Draw edges
    for edge in G.edges():
        arrowstyle = G.edges[edge]["arrowstyle"]
        nx.draw_networkx_edges(G, pos, node_size=1000,
                            edgelist = [edge],
                            arrowstyle=arrowstyle,
                            arrowsize=20,width=2)

    # Add labels to the nodes
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    # Set the title
    plt.title(f'Bidirectional Network (Time Step {t % 50}/50 [{"Minus" if ceil(t/50)%2 else "Plus"}], Sample {floor(t/100)%numSamples+1}/{numSamples}, Epoch {ceil(t/100/numSamples)}/{numEpochs})')
    # Add a color bar legend
    plt.colorbar(nodes, label='Neuron Activation (rate code)')

    
    # Save frame into bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format = "png", dpi = 150)
    buf.seek(0)
    animation_frames.append(buf)

# Convert frames to a GIF
output_gif = 'neural_network_activities.gif'
imageio.mimsave(output_gif,
                [imageio.imread(frame) for frame in animation_frames],
                duration=0.1,
                loop = 0)

# Clean up buffers
for buf in animation_frames:
    buf.close()


# Plot RMSE over time
plt.figure()
plt.plot(resultMixed[1:], label="Mixed")
baseline = np.mean([RMSE(entry, targets) for entry in np.random.uniform(size=(2000,numSamples,4))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")
plt.title("Iris Dataset")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()