from dataset import *
from gesture import *
from topology import *
import random

random.seed(0)

########## Emissions #########
## gesture_emissions
# Defines the emissions for the hmm
def gesture_emissions(n_states,  scale = 1):
    distributions = []
    step = scale / n_states;
    for i in range(0, n_states):
        a = scale - (i * step)
        b = scale - (i * step)
        gaussianX = NormalDistribution(a, scale * 0.01)
        gaussianY = NormalDistribution(b, scale * 0.01)
        distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY]))
    return  distributions

## create_hmm_gesture
# Defines the hmm for recognizing the specified gesture
def create_hmm_gesture(name, training_set, n_states = 8):
    # Creates hmm
    topology_factory = HiddenMarkovModelTopology()
    emissions = gesture_emissions(n_states, scale=100)
    model = topology_factory.forward(name, n_states, emissions)
    # Train
    model.fit(training_set, use_pseudocount=True)

    return model


## plot_gesture
# Plots the model input examples.
#@staticmethod
# Plot a model gesture
def plot_gesture(model):
    for i in range(1, 3):
        sequence = model.sample()
        result = numpy.array(sequence).astype('float')
        plt.axis("equal")
        plt.plot(result[:, 0], result[:, 1])
        plt.show()
