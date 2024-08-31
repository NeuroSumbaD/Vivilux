from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
import vivilux.photonics as px
from vivilux.photonics.meshes import MZImesh, DiagMZI, SVDMZI

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
np.random.seed(seed=0)


matrixSize = 4

record = {}
for meshType in [MZImesh, DiagMZI, SVDMZI]:

    magnitudes = np.linspace(-1,1,300)
    maxConcavity = 0.0
    record[meshType.NAME] = {"SSE": [],
                             "coeff": [],
                             }
    plt.figure()
    for iteration in range(200):
        dummyLayer = Layer(matrixSize, isInput=True, name="Input")
        dummyNet = Net(name = "LEABRA_NET")
        dummyNet.AddLayer(dummyLayer)
        mzi = meshType(matrixSize, dummyLayer,
                    numDirections=28,
                    numSteps=500,
                    rtol=5e-2,
                    updateMagnitude=1e-6,
                    )

        initialMatrix = mzi.get()/mzi.Gscale
        derivativeMatrix, stepVectors = mzi.matrixGradient()
        flatVectors = np.concatenate([stepVector.flatten() for stepVector in stepVectors])
        updateMagnitude = np.sqrt(np.sum(np.square(flatVectors)))
        unitStepVectors = [stepVector/updateMagnitude for stepVector in stepVectors]
        # assert(np.isclose(np.sqrt(np.sum(np.square(flatVectors))), 1, rtol=1e-3, atol=0))

        errors = []

        for updateMagnitude in magnitudes:
            expectedDelta = updateMagnitude * derivativeMatrix

            stepParams = [param + updateMagnitude*step for param, step in zip(mzi.getParams(), unitStepVectors)]
            mzi.boundParams(stepParams)
            trueDelta = mzi.getFromParams(stepParams) - initialMatrix

            SSE = np.sum(np.square(expectedDelta.flatten() - trueDelta.flatten()))

            errors.append(SSE)
        
        coeff = np.polyfit(magnitudes, errors, 2)
        record[meshType.NAME]["coeff"].append(coeff)
        record[meshType.NAME]["SSE"].append(errors)
        if iteration < 6:
            plt.plot(magnitudes, errors, label=str(iteration))

    record[meshType.NAME]["coeff"] = np.array(record[meshType.NAME]["coeff"])
    concavities = record[meshType.NAME]["coeff"][:,0]
    maxConcavity = np.max(concavities)
    mean = np.mean(concavities)
    std = np.std(concavities)
    print(f"Max concavity for [{meshType.NAME}]: {maxConcavity}")
    print(f"\tmean concavity: {mean}, std dev: {std}")

    # plt.axhline(1e-3, color="r", linestyle="--", label="")
    plt.title(f"Step magnitude vs derivative error [{meshType.NAME}]")
    plt.xlabel("magnitude of step")
    plt.ylabel("derivative error (SSE)")
    plt.legend()

for mesh in record:
    concavities = record[mesh]["coeff"][:,0]
    maxConcavity = concavities.max()
    minConcavity = concavities.min()
    plt.figure()
    plt.hist(concavities, bins=np.arange(minConcavity,maxConcavity+0.1,0.1))
    plt.title(f"Central difference approximation histogram [{mesh}]")
    plt.xlabel("concavity of error")
    plt.ylabel("# occurences")

plt.show()