from . import Layer, Mesh, Net, DELTA_TIME
from vivilux.activations import ReLu
from vivilux.learningRules import CHL
from vivilux.visualize import Monitor

import numpy as np

class IsingNet(Net):
    '''Ising machine class which has one recurrent Layer of some neuron type
        with bistable output.
    '''
    def __init__(self, layers: list[Layer], meshType: Mesh, name=None, meshArgs={}, numTimeSteps=50, monitoring=False, defMonitor=Monitor, **kwargs):
        '''Instantiates the mesh with a single layer as input and output.
        '''
        self.DELTA_TIME = DELTA_TIME
        self.numTimeSteps = numTimeSteps
        self.monitoring = monitoring
        self.defMonitor = defMonitor

        self.name =  f"ISING_NET_{Net.count}" if name == None else name
        Net.count += 1

        self.layers = layers

        for index, layer in enumerate(self.layers):
            size = len(layer)
            layer.addMesh(meshType(size, layer,
                                   **meshArgs))
            if monitoring:
                layer.monitor = self.defMonitor(name = self.name + ": " + layer.name,
                                    labels = ["time step", "activity"],
                                    limits=[numTimeSteps, 2],
                                    numLines=len(layer))
                
    def PhaseStep(self):
        phaseActivity = []
        for time in range(self.numTimeSteps):
            phaseActivity.append(self.TimeStep())
        return np.array(phaseActivity)

    def TimeStep(self):
        layerActivities = []
        for layer in self.layers:
            layerActivities.append(layer.TimeStep())
        return np.array(layerActivities).reshape(1, -1) #flatten all activity into single timestep

    def Run(self, numPhases, flatten=True):
        phases = []
        for step in range(numPhases):
            phases.append(self.PhaseStep())
        if flatten: #flatten phases into timesteps
            return np.array(phases).reshape(-1, np.sum([len(lay) for lay in self.layers]))
        else:
            return np.array(phases)



class RingOscillator(Layer):
    '''Defines a layer of neurons which operate using ring oscillator logic to
        generate a square wave. External input turns off the signal such that
        coupled oscillators will try to fire out of phase.

        offset is a number of timesteps (int) that offset the phase of an oscillator
    '''
    def __init__(self, length, activation=ReLu(), name=None, offset=[], EN=[], period=20):
        self.modified = False 
        self.act = activation
        self.offset = offset if len(offset) > 0 else np.zeros(length)
        self.EN = EN if len(EN) > 0 else np.ones(length).astype(bool)
        self.period = period

        self.monitor = None
        self.snapshot = {}

        # Initialize layer activities
        self.excAct = np.zeros(length) # linearly integrated dendritic inputs (internal Activation)
        self.inhAct = np.zeros(length)
        self.outAct = np.zeros(length).astype(bool)
        self.counter = np.array(self.offset) # counts steps since last flip
        self.modified = True
        # Empty initial excitatory and inhibitory meshes
        self.excMeshes: list[Mesh] = []
        self.inhMeshes: list[Mesh] = []
        self.EXT = np.ones(length).astype(bool)
        
        self.freeze = False
        self.name =  f"OSC_LAYER_{Layer.count}" if name == None else name

    def getActivity(self, modify = False):
        if self.modified == True or modify:
            self.modified = False
            self.excAct -= DELTA_TIME*self.excAct
            self.Integrate()
            DET = self.act(self.excAct) > 0.5 # TIA output above logical threshold
            EXT = np.logical_not(np.logical_and(DET, self.EN))
            fallingEdge = np.logical_and(self.EXT, np.logical_not(EXT))
            self.EXT = EXT
            # Calculate output activity
            boolAct = self.outAct.astype(bool)
            self.counter +=1
            flipMask = self.counter > np.floor(self.period/2)
            # reset counter each time falling edge or flip is true
            reset = np.logical_or(fallingEdge, flipMask)
            self.counter *= np.logical_not(reset) 
            internalOscillation = np.logical_xor(flipMask, boolAct)
            externalOscillation = np.logical_and(EXT, internalOscillation)
            self.outAct = externalOscillation

        return self.outAct
    
    def Integrate(self):
        self.excAct = np.zeros(len(self))
        self.inhAct = np.zeros(len(self))
        for mesh in self.excMeshes:
            self.excAct += mesh.apply()[:len(self)]

        for mesh in self.inhMeshes:
            self.inhAct += mesh.apply()[:len(self)]

    
    def setEN(self, EN: np.ndarray):
        self.EN = np.array(EN).astype(bool)
        self.modified = True

    def TimeStep(self):
        return self.getActivity(modify=True)