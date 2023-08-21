# ViviLux

A package for simulating meshes of photonic elements with neuromorphic learning rules
*more info coming soon*

## Local Install

For testing and development, clone and cd into repository and type:

~~~ 
pip install -e .
~~~

## Simulator

A network must be defined in terms of the types of layer pools (number of neurons, activation functions, etc.), synaptic meshes (ideal abstractions verus hardware implmentations). A *presentation* describes each instance that a stimulus is supplied to the network as input. A single presentation includes multiple *phases* of network activity in which neural mechanisms might turn on or off. A *trial* describes the full simulation of the network in response to a presentation.

The simulation execution is broken down into several intervals of time:
1. timeStep: smallest unit for temporal integration of all variables. Set using DELTA_TIME variable.
2. phaseStep: a subunit of time during a presentation in which certain mechanisms will be present in the network's simulation. This might represent the presence of stimulus at a particular layer, or some other process which is not meant to be run over the whole simulation.
3. trialStep: unit of time demarcating a *presentation* of a stimulus. This includes full simulation of all phases defined for the network.

## Learning Rules

### CHL

## Optimizers

## Activations