from dataset import *
from gesture import *
from topology import *


## Multistroke
######### Modelled Gesture #########
## Arrow
#
def create_arrow(primitiveDir, n_states):
    # Primitive hmms
    first = primitive_model(primitiveDir, n_states, direction=Primitive.diagonal, theta = 40)
    second = primitive_model(primitiveDir, n_states, direction=Primitive.diagonal, theta = -160)
    third = primitive_model(primitiveDir, n_states, direction=Primitive.diagonal, theta= 20)
    fourth = primitive_model(primitiveDir, n_states, direction=Primitive.diagonal, theta= -110)

    # Translation emission
    translationDistribution(second, [0, 20])
    translationDistribution(third, [0, 20])
    translationDistribution(fourth, [30, 0])

    # Link models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third, fourth])
    # Model name
    model.name = 'arrow'
    return model, seq


## Caret
# diagonal 60° + diagonal -60°
def create_caret(baseDir, n_states):
    # Primitive hmms
    first = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta = 60)
    second = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta = -60)

    # Translation emission
    translationDistribution(second, [50, 0])

    # Link models
    model, seq = HiddenMarkovModelTopology.sequence([first, second])
    # Model name
    model.name = 'caret'
    return model, seq

## Delete
# diagonal -45° + left + diagonal 45°
def create_delete(baseDir, n_states):
    # Primitive hmms
    first = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta=-45)
    second = primitive_model(baseDir, n_states, direction=Primitive.left)
    third = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta=45)

    translationDistribution(second, [0, -50])

    # Link models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    # Model name
    model.name = 'delete'
    return model, seq

## Rectangle
# left + right + up + down
def create_rectangle(baseDir, n_states):
    # Primitive hmms
    first = primitive_model(baseDir, n_states, direction=Primitive.down, theta=270)
    second = primitive_model(baseDir, n_states, direction=Primitive.right)
    third = primitive_model(baseDir, n_states, direction=Primitive.up, theta=90)
    fourth = primitive_model(baseDir, n_states, direction=Primitive.left)

    # Fix emission first
    translationDistribution(first, [-50, 0])
    translationDistribution(second, [0, -50])
    translationDistribution(third, [50, 0])
    translationDistribution(fourth, [0, 50])

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third, fourth])
    # Model name
    model.name = 'rectangle'
    return model, seq

## Square Braket Left
# left + down + right
def create_square_braket_left(baseDir, n_states):
    # Primitive hmms
    first = primitive_model(baseDir, n_states, direction=Primitive.left)
    second = primitive_model(baseDir, n_states, direction=Primitive.down, theta=270)
    third = primitive_model(baseDir, n_states, direction=Primitive.right)

    translationDistribution(first, [0, 40])
    translationDistribution(second, [-40, 0])
    translationDistribution(third, [0, -40])

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    # Model name
    model.name = 'left_sq_bracket'
    return model, seq

## Square Braket Right
# right + down + left
def create_square_braket_right(baseDir, n_states):
    # Creazione hmm primitive
    first = primitive_model(baseDir, n_states, direction=Primitive.right)
    second = primitive_model(baseDir, n_states, direction=Primitive.down, theta=270)
    third = primitive_model(baseDir, n_states, direction=Primitive.left)

    translationDistribution(first, [0, 40])
    translationDistribution(second, [40, 0])
    translationDistribution(third, [0, -40])

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    # Model name
    model.name = 'right_sq_bracket'
    return model, seq

## Star
# diagonal 60° + diagonal -60° + diagonal 150° + right + diagonal -150°
def create_star(baseDir, n_states):
    # Creazione hmm Primitive
    first = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta=70)
    second = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta=-70)
    third = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta=150)
    fourth = primitive_model(baseDir, n_states, direction=Primitive.right)
    fifth = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta=-150)

    translationDistribution(first, [-20, 0])
    translationDistribution(second, [20, 0])
    translationDistribution(third, [10, -20])
    translationDistribution(fifth, [0, -20])

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third, fourth, fifth])
    # Model name
    model.name = 'star'
    return model, seq

## Triangle
# diagonal -135° + right + diagonal 135°
def create_triangle(baseDir, n_states):
    # Creazione hmm Primitive
    first = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta = -120)
    second = primitive_model(baseDir, n_states, direction=Primitive.right)
    third = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta = 120)

    translationDistribution(first, [-30, 0])
    translationDistribution(second, [0, -40])
    translationDistribution(third, [30, 0])

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    # Model name
    model.name = 'triangle'
    return model, seq

## V
# diagonal -60° + diagonal 60°
def create_v(baseDir, n_states):
    # Creazione primitive
    first = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta = -60)
    second = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta = 60)

    translationDistribution(second, [50, 0])

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second])
    # Model name
    model.name = 'v'
    return model, seq

## X
# diagonal -45° + up + diagonal -135°
def create_x(baseDir, n_states):
    # Primitives
    first = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta = -45)
    second = primitive_model(baseDir, n_states, direction=Primitive.up, theta = 90)
    third = primitive_model(baseDir, n_states, direction=Primitive.diagonal, theta = -145)

    translationDistribution(second, [50, 0])

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    # Model name
    model.name = 'x'
    return model, seq


## TranslationDistribution
#
def translationDistribution2(model, cols=[0,1]):
    # Translation matrix
    n = len(cols)+1
    i,j = numpy.indices(matrix_translantion.shape)
    matrix_translantion[i == j] = 1
    matrix_translantion = numpy.zeros((n,n))
    for i in range(0, len(cols)):
        matrix_translantion[i, i, len(cols)] = cols[i]

    # Fix emission with translation
    for i in range(0, len(model.states)):
        state = model.states[i]
        if state != model.start and state != model.end:
            state.distribution = state.distribution * matrix_translantion

    return model

def translationDistribution(model, cols=[0,1]):
    # Translation matrix
    n = len(cols)+1
    matrix_translantion = numpy.zeros((n,n))
    i,j = numpy.indices(matrix_translantion.shape)
    matrix_translantion[i == j] = 1
    for i in range(0, len(cols)):
        matrix_translantion[i, len(cols)] = cols[i]

    # Fix emission with translation
    for index in range(0, len(model.states)):
        state = model.states[index]
        if state != model.start and state != model.end:
            emission = state.distribution
            # Changes x values
            x = emission.distributions[0].parameters[0] + cols[0]
            # Changes y values
            y = emission.distributions[1].parameters[0] + cols[1]
            # Makes new distribution
            gaussianX = NormalDistribution(x, emission.distributions[0].parameters[1])
            gaussianY = NormalDistribution(y, emission.distributions[1].parameters[1])
            state.distribution = IndependentComponentsDistribution([gaussianX, gaussianY])
    return model