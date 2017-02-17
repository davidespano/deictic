from dataset import *
from gesture import *
from topology import *
import random

random.seed(0)

######### Training #########
## training_leave_one_out
# Provides to train the input hmm (leave one out)
def train_hmm_gesture(model, sequences):
    # train the hmm
    model.fit(sequences, use_pseudocount=True)
    return model

########## Emissions #########
## gesture_emissions
# Defines the emissions for the hmm
def emission_hmm_gesture(n_states,  scale = 1):
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
def create_hmm_gesture(name, traning_sequences, n_states = 8, scale=100):

    # Creates hmm
    topology_factory = HiddenMarkovModelTopology()
    emissions = emission_hmm_gesture(n_states, scale)
    model = topology_factory.forward(name, n_states, emissions)

    # Training phase (leave one out). index_file is used for determines which file we will use during testing.
    model = train_hmm_gesture(model, traning_sequences)
    return model


## plot_gesture
# Plots the model input examples.
def plot_gesture(model):
    sequence = model.sample()
    result = numpy.array(sequence).astype('float')
    plt.axis("equal")
    plt.plot(result[:, 0], result[:, 1])
    plt.title(model.name)
    plt.show()
