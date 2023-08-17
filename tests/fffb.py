import vivilux as vl
import vivilux.photonics
from vivilux import FFFB, Layer, Mesh, RecurNet
from vivilux.learningRules import CHL, GeneRec, ByPass
from vivilux.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=0)

import pandas as pd
import seaborn as sns

from copy import deepcopy

numSamples = 40
numEpochs = 50

#define input and output data (must be normalized and positive-valued)
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
inputs = np.abs(vecs/mags[...,np.newaxis])
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
targets = np.abs(vecs/mags[...,np.newaxis])
del vecs, mags

df = pd.DataFrame(columns=["FF", "FB", "FBTau", "FF0"])

recurNet = RecurNet([
                    Layer(4, isInput=True),
                    Layer(4, learningRule=CHL),
                    Layer(4, learningRule=CHL)
                ], Mesh,
                learningRate = 0.1,
                name = f"Base_FFFBnet",
                )

fffbNet = FFFB([
                    Layer(4, isInput=True),
                    Layer(4, learningRule=CHL),
                    Layer(4, learningRule=CHL)
                ], Mesh,
                learningRate = 0.1,
                name = f"Base_FFFBnet",
                )

fffbNet_phot = FFFB([
                    vl.photonics.PhotonicLayer(4, isInput=True),
                    vl.photonics.PhotonicLayer(4, learningRule=CHL),
                    vl.photonics.PhotonicLayer(4, learningRule=CHL)
                ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
                learningRate = 0.1,
                name = f"Base_FFFBnet",
                )


resultRecur = recurNet.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
plt.plot(resultRecur, label="Recurrent Net")

resultFFFB = fffbNet.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
plt.plot(resultFFFB, label="FFFB Net")

resultFFFB_phot = fffbNet_phot.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
plt.plot(resultFFFB, label="FFFB Net (Photonic)")

plt.title("Random Input/Output Matching with Positive Weights")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")