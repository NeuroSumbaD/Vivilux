# ViviLux

ViviLux is a package for simulating and testing neuromorphic learning rules on various photonic hardware architectures. Our design philosophy is distinct from traditional deep learning and aims to provide tools for benchmarking and designing brain-like dynamical systems with online learning and hardware-efficient information encodings of information. The intention is to enable a new generation of SWaP (size, weight, and power) optimized computers based on biological brains!

The package is under active development and aims to provide the following features:
- Error-driven learning simulations based on dynamics local to each synapse (or synaptic mesh)
- Varying levels of simulation abstraction with algorithmic-, architecture-, and device-level accuracy
- Visualization modules for inspecting and debugging behavior

## Local Install

For testing and development, clone and cd into the repository and type:

~~~ 
pip install -e .
~~~


## Simulator

A network must be defined in terms of the types of layer pools (number of neurons, activation functions, etc.), synaptic meshes (ideal abstractions versus hardware implementations). A *presentation* describes each instance that a stimulus is supplied to the network as input. A single presentation includes multiple *phases* of network activity in which neural mechanisms might turn on or off. A *trial* describes the full simulation of the network in response to a presentation.

The simulation execution is broken down into several intervals of time:
1. timeStep: smallest unit for temporal integration of all variables. Set using DELTA_TIME variable.
2. phaseStep: a subunit of time during a presentation in which certain mechanisms will be present in the network's simulation. This might represent the presence of stimulus at a particular layer, or some other process that is not meant to be run over the whole simulation.
3. trialStep: unit of time demarcating a *presentation* of a stimulus. This includes full simulation of all phases defined for the network.

## Learning Rules

### CHL

## Optimizers

## Activations
