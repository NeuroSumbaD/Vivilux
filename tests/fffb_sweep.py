from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.paths import Path
from vivilux.metrics import ThrMSE, ThrSSE
from vivilux.visualize import StackedMonitor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
np.random.seed(seed=0)

from copy import deepcopy
import pathlib
from os import path
import json
import time

#TODO: REMOVE THIS LINE, USED FOR SUPPRESSING UNWANTED WARNINGS IN DEBUGGING
import warnings
warnings.filterwarnings("ignore")

from itertools import product
import time

directory = pathlib.Path(__file__).parent.resolve()

# Number of Hyperparameter Combinations: 155727 -> 25272 -> 13608 -> 7875
hyperparameter_values = {
    'Gi': np.arange(1.3, 2.6, 0.2), # 7
    'FF': np.arange(0, 1.1, 0.5), # 3
    'FB': np.arange(0, 1.1, 0.5), # 3
    'FBTau': np.array([1.4, 3, 5]), # 3
    'MaxVsAvg': np.array([0, 0.5, 1]), # 3
    'FF0': np.arange(0, 0.11, 0.05) # 3
}

layerSize = 4
inputs = np.array(list(product(*[np.arange(0, 1, 0.2)] * layerSize))) # 625 inputs
identityMesh = np.eye(layerSize)

inputLayer = Layer(layerSize, isInput=True, name="Input")
outputLayer = Layer(layerSize, isTarget=True, name="Output")
outputConfig = deepcopy(layerConfig_std)

df = pd.DataFrame()
start = time.time()
for Gi in hyperparameter_values['Gi']:
    for FF in hyperparameter_values['FF']:
        for FB in hyperparameter_values['FB']:
            for FBTau in hyperparameter_values['FBTau']:
                for MaxVsAvg in hyperparameter_values['MaxVsAvg']:
                    for FF0 in hyperparameter_values['FF0']:
                        print(f"Searching parameters: \n\tGi={Gi}, FF={FF}, FB={FB}, FBTau={FBTau}, MaxVsAvg={MaxVsAvg}, FF0={FF0}")
                        outputConfig["FFFBparams"]['Gi'] = Gi
                        outputConfig["FFFBparams"]['FF'] = FF
                        outputConfig["FFFBparams"]['FB'] = FB
                        outputConfig["FFFBparams"]['FBTau'] = FBTau
                        outputConfig["FFFBparams"]['MaxVsAvg'] = MaxVsAvg
                        outputConfig["FFFBparams"]['FF0'] = FF0

                        # For layerSize == 4, numInputs = 625
                        leabraNet = Net(name = "LEABRA_NET",
                                        monitoring= False,
                                        )
                        
                        leabraNet.AddLayer(inputLayer)
                        leabraNet.AddLayer(outputLayer, layerConfig=outputConfig)

                        mesh = leabraNet.AddConnection(inputLayer, outputLayer)
                        mesh.set(matrix=identityMesh)

                        # Number of Inferences: 1557270000
                        output_vectors = np.array(leabraNet.Infer(input=inputs, verbosity=0)['target'])
                        # print(output_vector)

                        inL0 = np.count_nonzero(inputs, axis=1)
                        inL1 = np.linalg.norm(inputs, ord=1, axis=1)
                        inL2 = np.linalg.norm(inputs, ord=2, axis=1)

                        outL0 = np.count_nonzero(output_vectors, axis=1)
                        outL1 = np.linalg.norm(output_vectors, ord=1, axis=1)
                        outL2 = np.linalg.norm(output_vectors, ord=2, axis=1)
                        

                        
                        currentEntry = {
                            "Gi": Gi,
                            "FF": FF,
                            "FB": FB,
                            "FBTau": FBTau,
                            "MaxVsAvg": MaxVsAvg,
                            "FF0": FF0,
                            # "input_vector": inputs,
                            # "output_vectors": output_vectors,
                            "inL0": inL0,
                            "inL1": inL1,
                            "inL2": inL2,
                            "outL0": outL0,
                            "outL1": outL1,
                            "outL2": outL2,
                        }
                        df = pd.concat([df, pd.DataFrame(currentEntry)])
end = time.time()
duration = end-start
hour = duration//(60*60)
min = (duration-hour*60*60)//(60)
sec = duration-hour*60*60-min*60
df.to_csv(path.join(directory, "fffb_sweep_results.csv"), index=False)
print(f"Took {hour}h {min}min {sec}s")
