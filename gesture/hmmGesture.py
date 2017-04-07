from dataset import *
from gesture import *
from topology import *
import random

random.seed(0)

########## Emissions #########
## gesture_emissions
# Defines the emissions for the hmm
def gesture_emissions(n_states,  n_features = 2, scale = 1, weights = None):
    distributions = []
    step = scale / n_states;
    for i in range(0, n_states):
        distr_list = []
        for i in range(0, n_features):
            value = scale - (i * step)
            gaussian = NormalDistribution(value, scale * 0.01)
            distr_list.append(gaussian)
        if weights is None:
            distributions.append(IndependentComponentsDistribution(distr_list))
        else:
            distributions.append(IndependentComponentsDistribution(distr_list, weights = weights))
    return  distributions

## create_hmm_gesture
# Defines the hmm for recognizing the specified gesture
def create_hmm_gesture(name, training_set, n_states = 8, n_features = 2, weights = None, stroke= 1):
    # Creates hmm
    topology_factory = HiddenMarkovModelTopology()

    #TODO eliminare questo orrore
    emissions = gesture_emissions(n_states, n_features = 2, scale=100, weights=weights)
    model = topology_factory.forward(name, n_states, emissions)
    # Train
    model.fit(training_set, use_pseudocount=True)

    sps = n_states / stroke
    if n_features == 3:
        for i in range(0, len(model.states)):
            state = model.states[i]
            if not state.distribution is None:
                x = state.distribution.distributions[0]
                y = state.distribution.distributions[1]
                # s = NormalDistribution(self.stroke + 1.0, (i +1 ) * step)
                s = NormalDistribution(math.trunc(i/sps) + 1.0, 0.01)
                state.distribution = IndependentComponentsDistribution([x, y, s], weights=[1, 1, 10000])
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
        plt.plot(result[:, 0], result[:, 1], marker=".")
        plt.show()
