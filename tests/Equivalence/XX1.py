from vivilux.activations import NoisyXX1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from os import path
import pathlib

act = NoisyXX1(Thr = 0.5,
	Gain = 30,
	NVar = 0.01,
	VmActThr = 0.01)


inAx = np.linspace(-1,1, 200)
outAx = act(inAx)

# Read in CSV
directory = pathlib.Path(__file__).parent.resolve()
df = pd.read_csv(path.join(directory, "xx1.csv"))

plt.plot(inAx, outAx, label="Vivilux")
plt.plot(df["input"], df["output"], label="leabra")
plt.title("NoisyXX1")
plt.legend()
plt.show()


emerIn = df["input"].to_numpy()
print("In equivalence: ", np.all(np.isclose(emerIn, inAx, rtol=1e-3, atol=1e-3)))

emerOut = df["output"].to_numpy()
print("Out equivalence: ", np.all(np.isclose(emerOut, outAx, rtol=1e-3, atol=1e-3)))