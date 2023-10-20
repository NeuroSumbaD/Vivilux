import vivilux as vl
from vivilux import ising
from vivilux import photonics

import numpy as np
import matplotlib.pyplot as plt

offsets = [1, 6, 7, 5]
EN = [0,0,0,0]

ringOscillator = ising.RingOscillator(4, offset=offsets, EN=EN)
isingMachine = ising.IsingNet([
    ringOscillator,
    ],
    # meshType=photonics.MZImesh,
    meshType=vl.Mesh,
    name="Ising Machine")

mesh = isingMachine.layers[0].excMeshes[0]
mesh.matrix = np.array([[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]])

plt.ioff()
results = isingMachine.Run(5)

size = len(ringOscillator)
tier = np.arange(size)
plt.plot(results + tier)
plt.title("Oscillation without external input")
plt.legend(tier)
plt.show()

isingMachine.layers[0].setEN([1,1,1,1])
results2 = isingMachine.Run(5)
plt.plot(results2 + tier)
plt.title("Oscillation with external input")
plt.legend(tier)
plt.show()
