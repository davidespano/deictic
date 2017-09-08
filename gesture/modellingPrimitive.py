from dataset import *
from topology import *
from gesture import *
import random
from gesture import *
from enum import Enum

random.seed(0)

################### NOT USED ###################
# Primitive
class Primitive(Enum):
    left = 3
    right = 1
    up = 2
    down = 4
    diagonal = 5
    forward = 6
    behind = 7

######### Model ##########
## model_primitive
# Creates and returns the hmm model for the specified primitive
def primitive_model(baseDir, n_states, direction, theta = 0):
    # Left
    if(direction == Primitive.left):
        name = 'left'
    # Right
    elif(direction == Primitive.right):
        name = 'right'
    # Diagonal
    else:
        name = str(theta)

    # Get dataset
    dataset = CsvDataset(baseDir + name + '/')
    # Create model
    model = create_model(direction, name, n_states)
    # Training
    model.fit(dataset.readDataset(), use_pseudocount=True)

    return model

## topology_model
# Defines the topology and the hmm model
def create_model(direction, name, n_states=8):
    # create the hmm model
    topology_factory = HiddenMarkovModelTopology() # Topology
    emissions = primitive_emissions(direction, n_states, scale=100) # Emissions
    return (topology_factory.forward(name, n_states, emissions))

######### Emissions #########
## model_emissions
# Creates the distributions of the primitive hmm model
def primitive_emissions(direction, n_states, dimension = 2,  scale = 1):
    distributions = []
    step = scale / n_states;
    # Random value for the normal distribution
    k = random.random() * scale;

    # 2D
    if dimension == 2:
        for i in range(0, n_states):
            if direction == Primitive.left:
                a = scale - (i * step)
                b = k
            elif direction == Primitive.right:
                a = i * step
                b = k
            elif direction == Primitive.up:
                a = k
                b = i * step
            elif direction == Primitive.down:
                a = k
                b = scale - (i * step)
            elif direction == Primitive.diagonal:
                a = i * step
                b = i * step

            gaussianX = NormalDistribution(a, scale * 0.01)
            gaussianY = NormalDistribution(b, scale * 0.01)
            distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY]))
    # 3D
    elif dimension == 3:
        for i in range(0, n_states):
            if direction == Primitive.forward:
                a = k
                b = k
                c = i * step
            elif direction == Primitive.behind:
                a = k
                b = k
                c = scale - (i * step)

            gaussianX = NormalDistribution(a, scale * 0.01)
            gaussianY = NormalDistribution(b, scale * 0.01)
            gaussianZ = NormalDistribution(c, scale * 0.01)
            distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY, gaussianZ]))

    return distributions