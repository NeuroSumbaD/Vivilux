# Documentation

## Net

A `Net` is a collection of neural network `Layers` and configuration parameters which define in what sequence various inputs are applied to layers within the network.

## Layer

A `Layer` is an abstraction representing a set of neurons. Layer objects are responsible for applying dynamic nonlinearities within the neural network. A dynamic system is formed based on the processes within 

### Processes

Each layer transforms its inputs based on a number of processes. Processes are organized into two temporal domains, `Cyclic` and `Phasic` which correspond to once each time step or once each phase.

## Path

A `Path` is an abstraction representing a set of synaptic connections between two layers and parameters which define how the synaptic variables are updated during training.