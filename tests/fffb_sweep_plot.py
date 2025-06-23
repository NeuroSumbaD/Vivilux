import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

import pathlib
from os import path

directory = pathlib.Path(__file__).parent.resolve()
df = pd.read_csv(path.join(directory,"fffb_sweep_results.csv"))

df = df[df["Gi"] == 1.3]
# df = df[df["FF"] == 1]
# df = df[df["FB"] == 1]
# df = df[df["FBTau"] == 1.4]
df = df[df["MaxVsAvg"] == 0]
df = df[df["FF0"] == 0.1]

plt.figure()
plt.scatter(df["inL1"], df["outL1"])
plt.title("Input Norm vs Output Norm")
plt.xlabel("inL1")
plt.ylabel("outL1")
plt.show()