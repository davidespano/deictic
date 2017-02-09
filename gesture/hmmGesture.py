from dataset import *
from gesture import *
from topology import *
import random

random.seed(0)

######### Training #########
## training_leave_one_out
# Provides to train the input hmm (leave one out)
def training_leave_one_out(model, baseDir, index, dimensions = 2):
    baseDir = baseDir+'/'
    # load dataset with correct examples and apply LOO technique
    correct = ToolsDataset(baseDir)
    if(index >= len(correct.getDatasetIterator().filenames)):
        index = len(correct.getDatasetIterator().filenames) - 1

    one, sequences = correct.leave_one_out(index, dimensions=dimensions, scale=100)
    # train the hmm
    model.fit(sequences, use_pseudocount=True)
    return model

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
def create_hmm_gesture(baseDir, name, index_file, n_states = 8, dimensions = 2):
    # Get dataset
    folder = baseDir + name
    # Creates hmm
    topology_factory = HiddenMarkovModelTopology()
    emissions = gesture_emissions(n_states, scale=100)
    model = topology_factory.forward(name, n_states, emissions)

    # Training phase (leave one out). index_file is used for determines which file we will use during testing.
    model = training_leave_one_out(model, folder, index_file, dimensions=dimensions)
    return model


## plot_gesture
# Plots the model input examples.
@staticmethod
# Plot a model gesture
def plot_gesture(model):
    for i in range(1, 3):
        sequence = model.sample()
        result = numpy.array(sequence).astype('float')
        plt.axis("equal")
        plt.plot(result[:, 0], result[:, 1])
        plt.show()
