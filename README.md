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
1. timeStep: smallest unit for temporal integration of all variables. Set using `DELTA_TIME` variable.
2. phaseStep: a subunit of time during a presentation in which certain mechanisms will be present in the network's simulation. This might represent the presence of stimulus at a particular layer, or some other process that is not meant to be run over the whole simulation.
3. trialStep: unit of time demarcating a *presentation* of a stimulus. This includes full simulation of all phases defined for the network.


## Processes

A `Process` is defined to be a high-level class with methods to manipulate the `Neuron` in some way. ViviLux defines two types of processes:

1. `NeuralProcess`: A base class for various high-level processes which generate a current stimulus to a neuron.
2. `PhasicProcess`: A base class for various high-level processes which affect the neuron in
        some structural aspect such as learning, pruning, etc.

### Neural Processes

- `FFFB`: A process which runs the FFFB (feedforward feedback) inhibitory mechanism developed by Prof.
        O'Reilly. This mechanism acts as lateral inhibitory neurons within a
        pool and produces sparse activity without requiring the additional 
        neural units and inhibitory synapses. There is both a feedforward component which inhibits large jumps in activity across a pool, and a feeback component which more dynamically smooths the activity over time.

### Phasic Processes
- `ActAvg`: A process for calculating average neuron activities for learning.
        Coordinates with the XCAL process to perform learning.

- `XCAL`: "Check-mark" linearized BCM-style learning rule which
            describes the calcium concentration versus change in a synaptic
            efficacy curve. This is proportional to change in weight strength
            versus the activity of sending and receiving neuron for a single
            synapse. `XCAL` is discussed in further depth in the "Learning Rules" section.

## Learning Rules

The networks defined with ViviLux are Bidirectional Recurrent Neural Networks (BRNNs). Learning occurs in two phases, the *minus phase* (expectation) and the *plus phase* (outcome). In the minus phase, the network runs an inference on the inputs and develops an output, or an expectation for an outcome. In the plus phase, the instantaneous or most-recent neural activity is taken as the actual outcome. Comparison between the plus and minus phases and the corresponding synaptic weight changes results in learning.

### CHL

Contrastive Hebbian Learning (CHL) modifies the synaptic weights between neurons based on the mathematical difference between their plus phase activity and the minus phase activity.

The CHL equation is defined as follows: $$\Delta w = (x^+ y^+) - (x^- y^-)$$

- $x$ is the sending neuron's activity
- $y$ is the receiving neuron's activity
- $\Delta w$ is the change in the synaptic weight
- $+$ regards the plus phase
- $-$ regards the minus phase

### XCAL 

The XCAL (eXtended Contrastive Attractor Learning Model) dWt function builds upon CHL with the aim to be more biologically realistic. In contrast to CHL, XCAL also achieves the following:

1. When the observed outcome neural activity is zero, the synaptic weight changes are zero.
2. Weight changes are based on average activations across periods of time, rather than instantaneous activations, which results in better representations of the final state or actual outcome.

The XCAL dWt function is a linear piecewise function with the visual form of a check-mark, defined as follows: 
$$f_{xcal}(xy, \theta_p) = xy-\theta_p, \, xy > \theta_p \theta_d \\ -xy(1- \theta_d), \, \, \, \, \text{   otherwise} $$ 

- $\Delta w$ is the change in synaptic weight
- $\theta_p$ is a dynamically changing threshold that marks the point at which $\Delta w$ switches from negative to positive
- $\theta_d$ marks where the slope of $\Delta w$ switches from negative to positive (typically, $\theta_d = 0.1$)
- $xy$ is the neural activity (product of sending neuron's activity and receiving neuron's activity)

### Error-Driven and Self-Organizing Learning

To achieve error-driven learning, the XCAL dWt function is used in combination with a dynamic time-averaged threshold for $\theta_p$. $\theta_p$ is defined as the medium-time scale average synaptic activity, obtained from the neural states during the minus phase. The most recent activity averages during the plus phase are used for the $xy$ term in the XCAL function.

Here is the error-driven learning function: $$\Delta w = f_{xcal}(x_sy_s, x_my_m)$$

- $x_sy_s$ is the short-term neural activity (reflecting actual outcome during the plus phase)
- $x_my_m$ is the medium-time scale average synaptic activity (reflecting expectation during the minus phase)

This can be combined with another term that takes into account the *long-term average activity of the postsynaptic neuron*, $y_l$. $y_l$ The computation to determine $y_l$ can be outlined in pseudocode ($y$ is the receiving neuron's activity):

- if $y > 0.2$, then $y_l = y_l + \frac{1}{\tau_1}(max-y_l)$
- else, $y_l = y_l + \frac{1}{\tau_1}(min-y_l)$
- $\tau_1$ is 10 by default (integrating over 10 trials)

Using $y_l$ as the dynamic threshold in the XCAL function ($\Delta w = f_{xcal}(xy, y_l)$) results in *self-organizing learning*. By combining the error-driven learning equation with the self-organizing model based on a long-term time average, we obtain a versatile learning function: $$\Delta w = \lambda_l f_{xcal}(x_sy_s, y_l) + \lambda_m f_{xcal}(x_sy_s, x_my_m)$$

- $\lambda_l$ is the learning rate for self-organizing learning
- $\lambda_m$ is the learning rate for error-driven learning
- $y_l$ is the long-term average activity of the postsynaptic neuron


## Optimizers

## Activations
