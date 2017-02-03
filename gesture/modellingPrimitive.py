from dataset import *
from topology import *
from gesture import *
import random
from gesture import *
from enum import Enum

random.seed(0)

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
## topology_model
# Defines the topology and the hmm model
def create_model(direction: object, n_states: object, name: object) -> object:
    # create the hmm model
    topology_factory = HiddenMarkovModelTopology() # Topology
    emissions = primitive_emissions(n_states, direction, scale=100) # Emissions
    return topology_factory.forward(name, n_states, emissions)

## model_primitive
# Creates and returns the hmm model for the specified primitive
def primitive_model(baseDir, n_states, direction, degree = 0):
    # Left, Right, Up, Down
    baseDir = baseDir + 'down-trajectory/'

    # Left
    if(direction == Primitive.left):
        name = 'left'
    # Right
    elif(direction == Primitive.right):
        name = 'right'
    # Up
    elif(direction == Primitive.up):
        name = 'up'
    # Down
    elif(direction == Primitive.down):
        name = 'down'
    # Diagonal
    elif (direction == Primitive.diagonal):
        name = 'diagonal_{}'.format(degree)
    # 3D
    elif direction == Primitive.forward or direction == Primitive.behind:
        # Forward
        if direction == Primitive.forward:
            name = 'forward'
        # Behind
        else:
            name = 'behind'
            #model = create_swipe_model(direction, n_states)
            #model.fit(dataset.read_dataset(3, 100), use_pseudocount=True)

    # Get dataset
    dataset = ToolsDataset(baseDir + name + '/')
    # Create model
    model = create_model(direction, n_states, name)
    # Training
    model.fit(dataset.read_dataset(2, 100), use_pseudocount=True)

    return model



######### Emissions #########
## model_emissions
# Creates the distributions of the primitive hmm model
def primitive_emissions(n_states, direction, dimension = 2,  scale = 1):
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