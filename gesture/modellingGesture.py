from dataset import *
from gesture import *
from topology import *

######### Modelled Gesture #########
## Caret
# diagonal 60° + diagonal -60°
def create_caret(baseDir, n_states):

    # Primitive hmms
    first = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree = 60)
    second = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree = -60)

    # Fix emission second model
    for i in range(0, len(second.states)):
        state = second.states[i]
        if state != second.start and state != second.end:
            e1 = state.distribution
            # Changes x values
            x = e1.distributions[0].parameters[0] + 40
            # Create new distribution
            gaussianX = NormalDistribution(x, e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(e1.distributions[1].parameters[0], e1.distributions[1].parameters[1])
            second.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Link models
    model, seq = HiddenMarkovModelTopology.sequence([first, second])
    # Model name
    model.name = 'caret'
    return model, seq

## Delete
# diagonal -45° + left + diagonal 45°
def create_delete(baseDir, n_states):
    # Primitive hmms
    first = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree=-45)
    second = primitive_model(baseDir, n_states, direction=Primitive.left)
    third = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree=45)

    # Link models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    # Model name
    model.name = 'delete'
    return model, seq

## Rectangle
# left + right + up + down
def create_rectangle(baseDir, n_states):
    # Primitive hmms
    first = primitive_model(baseDir, n_states, direction=Primitive.down)
    second = primitive_model(baseDir, n_states, direction=Primitive.right)
    third = primitive_model(baseDir, n_states, direction=Primitive.up)
    fourth = primitive_model(baseDir, n_states, direction=Primitive.left)

    # Fix emission
    for i in range(0, len(third.states)):
        state = third.states[i]
        if state != third.start and state != third.end:
            e1 = state.distribution
            # Changes x values
            c = e1.distributions[0].parameters[0] + 100
            # Makes new distribution
            gaussianX = NormalDistribution(c, e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(e1.distributions[1].parameters[0], e1.distributions[1].parameters[1])
            third.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Fix emission left
    for i in range(0, len(fourth.states)):
        state = fourth.states[i]
        if state != fourth.start and state != fourth.end:
            e1 = state.distribution
            # Changes y values
            c = e1.distributions[1].parameters[0] + 100
            # Makes new distribution
            gaussianX = NormalDistribution(e1.distributions[0].parameters[0], e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(c, e1.distributions[1].parameters[1])
            fourth.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

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
    second = primitive_model(baseDir, n_states, direction=Primitive.down)
    third = primitive_model(baseDir, n_states, direction=Primitive.right)

    # Fix emission First
    for i in range(0, len(first.states)):
        state = first.states[i]
        if state != first.start and state != first.end:
            e1 = state.distribution
            # Changes y values
            y = e1.distributions[1].parameters[0] + 100
            # Makes new distribution
            gaussianX = NormalDistribution(e1.distributions[0].parameters[0], e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(y, e1.distributions[1].parameters[1])
            first.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    # Model name
    model.name = 'square-braket-left'
    return model, seq

## Square Braket Right
# right + down + left
def create_square_braket_right(baseDir, n_states):
    # Creazione hmm primitive
    first = primitive_model(baseDir, n_states, direction=Primitive.right)
    second = primitive_model(baseDir, n_states, direction=Primitive.down)
    third = primitive_model(baseDir, n_states, direction=Primitive.left)

    # Fix emission First
    for i in range(0, len(first.states)):
        state = first.states[i]
        if state != first.start and state != first.end:
            e1 = state.distribution
            # Changes y values
            y = e1.distributions[1].parameters[0] + 100
            # Makes new distribution
            gaussianX = NormalDistribution(e1.distributions[0].parameters[0], e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(y, e1.distributions[1].parameters[1])
            first.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])
    # Fix emission Second
    for i in range(0, len(second.states)):
        state = second.states[i]
        if state != second.start and state != second.end:
            e1 = state.distribution
            # Changes x values
            x = e1.distributions[0].parameters[0] + 100
            # Makes new distribution
            gaussianX = NormalDistribution(x, e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(e1.distributions[1].parameters[0], e1.distributions[1].parameters[1])
            second.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    # Model name
    model.name = 'square-braket-right'
    return model, seq

## Star
# diagonal 60° + diagonal -60° + diagonal 150° + right + diagonal -150°
def create_star(baseDir, n_states):
    # Creazione hmm Primitive
    first = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree=60)
    second = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree=-60)
    third = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree=150)
    fourth = primitive_model(baseDir, n_states, direction=Primitive.right)
    fifth = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree=-150)

    # Fix emission Second
    for i in range(0, len(second.states)):
        state = second.states[i]
        if state != second.start and state != second.end:
            e1 = state.distribution
            # Changes x values
            x = e1.distributions[0].parameters[0] + 50
            # Makes new distribution
            gaussianX = NormalDistribution(x, e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(e1.distributions[1].parameters[0], e1.distributions[1].parameters[1])
            second.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])
    # Fix emission Fourth
    for i in range(0, len(fourth.states)):
        state = fourth.states[i]
        if state != fourth.start and state != fourth.end:
            e1 = state.distribution
            # Changes y values
            y = e1.distributions[1].parameters[0] + 50
            # Makes new distribution
            gaussianX = NormalDistribution(e1.distributions[0].parameters[0], e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(y, e1.distributions[1].parameters[1])
            fourth.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third, fourth, fifth])
    # Model name
    model.name = 'star'
    return model, seq

## Triangle
# diagonal -135° + right + diagonal 135°
def create_triangle(baseDir, n_states):
    # Creazione hmm Primitive
    first = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree = -135)
    second = primitive_model(baseDir, n_states, direction=Primitive.right)
    third = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree = 135)

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    # Model name
    model.name = 'triangle'
    return model, seq

## V
# diagonal -60° + diagonal 60°
def create_v(baseDir, n_states):
    # Creazione primitive
    first = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree = -60)
    second = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree = 60)

    # Fix emission second
    for i in range(0, len(second.states)):
        state = second.states[i]
        if state != second.start and state != second.end:
            e1 = state.distribution
            # Changes x values
            c = e1.distributions[0].parameters[0] + 45
            # Makes new distribution
            gaussianX = NormalDistribution(c, e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(e1.distributions[1].parameters[0], e1.distributions[1].parameters[1])
            second.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second])
    # Model name
    model.name = 'v'
    return model, seq

## X
# diagonal -45° + up + diagonal -135°
def create_x(baseDir, n_states):
    # Primitives
    first = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree = -45)
    second = primitive_model(baseDir, n_states, direction=Primitive.up)
    third = primitive_model(baseDir, n_states, direction=Primitive.diagonal, degree = -135)

    # Fix emission second
    for i in range(0, len(second.states)):
        state = second.states[i]
        if state != second.start and state != second.end:
            e1 = state.distribution
            # Changes x values
            x = e1.distributions[0].parameters[0] + 100
            # Changes y values
            y = e1.distributions[1].parameters[0]
            # Makes new distribution
            gaussianX = NormalDistribution(x, e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(y, e1.distributions[1].parameters[1])
            second.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Links models
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    # Model name
    model.name = 'x'
    return model, seq