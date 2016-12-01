from dataset import *
from pomegranate import *
from topology import *
import matplotlib.pyplot as plt
import random
from gesture import *
from enum import Enum

random.seed(0)

class Direction(Enum):
    left = 3
    right = 1
    up = 2
    down = 4
    diagonal = 5
    forward = 6
    behind = 7

class TypeTest(Enum):
    caret = 3
    delete = 11
    left_swipe = 9
    rectangle = 1
    right_swipe = 10
    square_braket_left = 6
    square_braket_right = 7
    star = 8
    triangle = 2
    v = 4
    x = 5

def assign(typeTest):
    if (typeTest == TypeTest.caret):
        nome = 'caret'
    elif (typeTest == TypeTest.delete):
        nome = 'delete'
    elif (typeTest == TypeTest.left_swipe):
        nome = 'left'
    elif (typeTest == TypeTest.rectangle):
        nome = 'rectangle'
    elif (typeTest == TypeTest.right_swipe):
        nome = 'right'
    elif (typeTest == TypeTest.square_braket_left):
        nome = 'square-braket-left'
    elif (typeTest == TypeTest.square_braket_right):
        nome = 'square-braket-right'
    elif (typeTest == TypeTest.star):
        nome = 'star'
    elif (typeTest == TypeTest.triangle):
        nome = 'triangle'
    elif (typeTest == TypeTest.v):
        nome = 'v'
    elif (typeTest == TypeTest.x):
        nome = 'x'

    return nome

# Training
def training_leave_one_out(model, correct_dir, index, dimensions = 2):
    # load dataset with correct examples and apply LOO technique
    correct = LeapDataset(correct_dir)
    if(index >= len(correct.getCsvDataset().filenames)):
        index = len(correct.getCsvDataset().filenames)-1

    one, sequences = correct.leave_one_out(index, dimensions=dimensions, scale=100)
    # train the hmm
    model.fit(sequences, use_pseudocount=True)
    return model

# Hmm definition
def create_gesture_emissions(n_states, scale=1):
    distributions = []
    step = scale / n_states
    for i in range(0, n_states):
        a = i * step
        b = i * step
        gaussianX = NormalDistribution(a, scale * 0.01)
        gaussianY = NormalDistribution(b, scale * 0.01)
        distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY]))
    return distributions

def create_swipe_emissions(n_states, direction=Direction.left, scale=1):
    distributions = []
    step = scale / n_states;
    k = random.random() * scale;
    for i in range(0, n_states):
        if direction == Direction.left:
            a = scale - (i * step)
            b = k
        elif direction == Direction.right:
            a = i * step
            b = k
        elif direction == Direction.up:
            a = k
            b = i * step
        elif direction == Direction.down:
            a = k
            b = scale - (i * step)
        elif direction == Direction.diagonal:
            a = i * step
            b = i * step

        gaussianX = NormalDistribution(a, scale * 0.01)
        gaussianY = NormalDistribution(b, scale * 0.01)
        distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY]))
    return distributions

def create_3d_swipe_emissions(n_states, direction = Direction.forward, scale = 1):
    distributions = []
    step = scale / n_states;
    k = random.random() * scale;
    for i in range(0, n_states):
        if direction == Direction.forward:
            a = k
            b = k
            c = i * step
        elif direction == Direction.behind:
            a = k
            b = k
            c = scale - (i * step)

        gaussianX = NormalDistribution(a, scale * 0.01)
        gaussianY = NormalDistribution(b, scale * 0.01)
        gaussianZ = NormalDistribution(c, scale * 0.01)
        distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY, gaussianZ]))
    return distributions

def create_complete_emissions(n_states, scale=1):
    distributions = []
    step = scale / n_states;
    for i in range(0, n_states):
        a = random.random() * scale
        b = random.random() * scale
        gaussianX = NormalDistribution(a, scale * 0.1)
        gaussianY = NormalDistribution(b, scale * 0.1)
        distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY]))
    return distributions

def create_3d_swipe_model(direction: object, n_states: object, name: object = None) -> object:
    # create the hmm model
    topology_factory = HiddenMarkovModelTopology()
    emissions = create_3d_swipe_emissions(n_states, direction, scale=100)
    if name is None:
        if direction == Direction.forward:
            name = 'forward'
        elif direction == Direction.behind:
            name = 'behind'

    swipe = topology_factory.forward(name, n_states, emissions)
    return swipe

def create_swipe_model(direction: object, n_states: object, name: object = None) -> object:
    # create the hmm model
    topology_factory = HiddenMarkovModelTopology()
    emissions = create_swipe_emissions(n_states, direction, scale=100)
    if name is None:
        if direction == Direction.left:
            name = 'left'
        elif direction == Direction.right:
            name = 'right'
        elif direction == Direction.up:
            name = 'up'
        elif direction == Direction.down:
            name = 'down'
        elif direction == Direction.diagonal:
            name = 'diagonal'
        #elif direction == Direction.diagonal_down:
            #name = 'diagonal_down-swipe'

    swipe = topology_factory.forward(name, n_states, emissions)
    return swipe

def create_complete_model(n_states, name=None):
    # Creazione hmm
    topology_factory = HiddenMarkovModelTopology()
    emissions = create_swipe_emissions(n_states, scale=100)
    swipe = topology_factory.forward(name, n_states, emissions)
    return swipe

#### Gesture ####
def create_gesture(typeTest, baseDir, n_states):
    seq = []
    nome = assign(typeTest)
    # Creazione modello
    if (typeTest == TypeTest.caret):
        model, seq = create_caret(baseDir, n_states)

    elif (typeTest == TypeTest.delete):
        model, seq = create_delete(baseDir, n_states)

    elif (typeTest == TypeTest.left_swipe):
        model = create_primitive_model(baseDir, n_states, direction = Direction.left)

    elif (typeTest == TypeTest.rectangle):
        model, seq = create_rectangle(baseDir, n_states)

    elif (typeTest == TypeTest.right_swipe):
        model = create_primitive_model(baseDir, n_states, direction = Direction.right)

    elif (typeTest == TypeTest.square_braket_left):
        model, seq = create_square_braket_left(baseDir, n_states)

    elif (typeTest == TypeTest.square_braket_right):
        model, seq = create_square_braket_right(baseDir, n_states)

    elif (typeTest == TypeTest.star):
        model, seq = create_star(baseDir, n_states)

    elif (typeTest == TypeTest.triangle):
        model, seq = create_triangle(baseDir, n_states)

    elif (typeTest == TypeTest.v):
        model, seq = create_v(baseDir, n_states)

    elif (typeTest == TypeTest.x):
        model, seq = create_x(baseDir, n_states)

    # Restituisce modello
    return model, seq

# Caret
def create_caret(baseDir, n_states):
    # Creazione hmm Primitive
    first = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree = 60)
    second = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree = -60)

    # Fix emission second
    for i in range(0, len(second.states)):
        state = second.states[i]
        if state != second.start and state != second.end:
            e1 = state.distribution
            # Vogliamo modificare la x
            x = e1.distributions[0].parameters[0] + 40
            # Crea nuova distribuzione
            gaussianX = NormalDistribution(x, e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(e1.distributions[1].parameters[0], e1.distributions[1].parameters[1])
            second.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Creazione gesture completa
    model, seq = HiddenMarkovModelTopology.sequence([first, second])
    model.name = 'caret'
    return model, seq

# Delete
def create_delete(baseDir, n_states):
    # Creazione hmm Primitive
    first = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree=-45)
    second = create_primitive_model(baseDir, n_states, direction=Direction.left)
    third = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree=45)

    # Creazione gesture completa
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    model.name = 'delete'
    return model, seq

# Rectangle
def create_rectangle(baseDir, n_states):
    # Creazione hmm primitive
    left = create_primitive_model(baseDir, n_states, direction=Direction.left)
    right = create_primitive_model(baseDir, n_states, direction=Direction.right)
    up = create_primitive_model(baseDir, n_states, direction=Direction.up)
    down = create_primitive_model(baseDir, n_states, direction=Direction.down)

    # Fix emission up
    for i in range(0, len(up.states)):
        state = up.states[i]
        if state != up.start and state != up.end:
            e1 = state.distribution
            # Vogliamo modificare la x
            c = e1.distributions[0].parameters[0] + 100
            # Crea nuova distribuzione
            gaussianX = NormalDistribution(c, e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(e1.distributions[1].parameters[0], e1.distributions[1].parameters[1])
            up.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Fix emission left
    for i in range(0, len(left.states)):
        state = left.states[i]
        if state != left.start and state != left.end:
            e1 = state.distribution
            # Vogliamo modificare la y
            c = e1.distributions[1].parameters[0] + 100
            # Crea nuova distribuzione
            gaussianX = NormalDistribution(e1.distributions[0].parameters[0], e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(c, e1.distributions[1].parameters[1])
            left.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Creazione hmm gesture completa
    model, seq = HiddenMarkovModelTopology.sequence([down, right, up, left])
    model.name = 'rectangle'
    return model, seq

# Square Braket Left
def create_square_braket_left(baseDir, n_states):
    # Creazione hmm primitive
    first = create_primitive_model(baseDir, n_states, direction=Direction.left)
    second = create_primitive_model(baseDir, n_states, direction=Direction.down)
    third = create_primitive_model(baseDir, n_states, direction=Direction.right)

    # Fix emission First
    for i in range(0, len(first.states)):
        state = first.states[i]
        if state != first.start and state != first.end:
            e1 = state.distribution
            # Vogliamo modificare la y
            y = e1.distributions[1].parameters[0] + 100
            # Crea nuova distribuzione
            gaussianX = NormalDistribution(e1.distributions[0].parameters[0], e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(y, e1.distributions[1].parameters[1])
            first.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Creazione hmm gesture completa
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    model.name = 'square-braket-left'
    return model, seq

# Square Braket Right
def create_square_braket_right(baseDir, n_states):
    # Creazione hmm primitive
    first = create_primitive_model(baseDir, n_states, direction=Direction.right)
    second = create_primitive_model(baseDir, n_states, direction=Direction.down)
    third = create_primitive_model(baseDir, n_states, direction=Direction.left)

    # Fix emission First
    for i in range(0, len(first.states)):
        state = first.states[i]
        if state != first.start and state != first.end:
            e1 = state.distribution
            # Vogliamo modificare la Y
            y = e1.distributions[1].parameters[0] + 100
            # Crea nuova distribuzione
            gaussianX = NormalDistribution(e1.distributions[0].parameters[0], e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(y, e1.distributions[1].parameters[1])
            first.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])
    # Fix emission Second
    for i in range(0, len(second.states)):
        state = second.states[i]
        if state != second.start and state != second.end:
            e1 = state.distribution
            # Vogliamo modificare la x
            x = e1.distributions[0].parameters[0] + 100
            # Crea nuova distribuzione
            gaussianX = NormalDistribution(x, e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(e1.distributions[1].parameters[0], e1.distributions[1].parameters[1])
            second.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Creazione hmm gesture completa
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    model.name = 'square-braket-right'
    return model, seq

# Star
def create_star(baseDir, n_states):
    # Creazione hmm Primitive
    first = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree=60)
    second = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree=-60)
    third = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree=150)
    fourth = create_primitive_model(baseDir, n_states, direction=Direction.right)
    fifth = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree=-150)

    # Fix emission Second
    for i in range(0, len(second.states)):
        state = second.states[i]
        if state != second.start and state != second.end:
            e1 = state.distribution
            # Vogliamo modificare la x
            x = e1.distributions[0].parameters[0] + 50
            # Crea nuova distribuzione
            gaussianX = NormalDistribution(x, e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(e1.distributions[1].parameters[0], e1.distributions[1].parameters[1])
            second.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])
    # Fix emission Fourth
    for i in range(0, len(fourth.states)):
        state = fourth.states[i]
        if state != fourth.start and state != fourth.end:
            e1 = state.distribution
            # Vogliamo modificare la y
            y = e1.distributions[1].parameters[0] + 50
            # Crea nuova distribuzione
            gaussianX = NormalDistribution(e1.distributions[0].parameters[0], e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(y, e1.distributions[1].parameters[1])
            fourth.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Creazione hmm gesture completa
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third, fourth, fifth])
    model.name = 'star'
    return model, seq


# Triangle
def create_triangle(baseDir, n_states):
    # Creazione hmm Primitive
    first = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree = -135)
    second = create_primitive_model(baseDir, n_states, direction=Direction.right)
    third = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree = 135)

    #diagonal_down_45 = create_primitive_model(baseDir, n_states, direction=Direction.diagonal_down, degree = 45)
    #diagonal_up_m_45 = create_primitive_model(baseDir, n_states, direction=Direction.diagonal_up, degree = 135)

    # Creazione hmm gesture completa
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    model.name = 'triangle'
    return model, seq

# V
def create_v(baseDir, n_states):
    # Creazione primitive
    first = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree = -60)
    second = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree = 60)

    # Fix emission second
    for i in range(0, len(second.states)):
        state = second.states[i]
        if state != second.start and state != second.end:
            e1 = state.distribution
            # Vogliamo modificare la x
            c = e1.distributions[0].parameters[0] + 45
            # Crea nuova distribuzione
            gaussianX = NormalDistribution(c, e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(e1.distributions[1].parameters[0], e1.distributions[1].parameters[1])
            second.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Creazione hmm gesture completa
    model, seq = HiddenMarkovModelTopology.sequence([first, second])
    model.name = 'v'
    return model, seq

# X
def create_x(baseDir, n_states):
    # Creazione primitive
    first = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree = 60)
    second = create_primitive_model(baseDir, n_states, direction=Direction.down)
    third = create_primitive_model(baseDir, n_states, direction=Direction.diagonal, degree = 135)

    # Fix emission second
    for i in range(0, len(second.states)):
        state = second.states[i]
        if state != second.start and state != second.end:
            e1 = state.distribution
            # Vogliamo modificare la x
            x = e1.distributions[0].parameters[0] + 80
            # Vogliamo modificare la y
            y = e1.distributions[1].parameters[0] - 20
            # Crea nuova distribuzione
            gaussianX = NormalDistribution(x, e1.distributions[0].parameters[1])
            gaussianY = NormalDistribution(y, e1.distributions[1].parameters[1])
            second.states[i].distribution = IndependentComponentsDistribution([gaussianX, gaussianY])

    # Creazione hmm gesture completa
    model, seq = HiddenMarkovModelTopology.sequence([first, second, third])
    model.name = 'x'
    return model, seq

# Gesture complete
def create_hmm_gesture_complete(nome, baseDir, n_states, index, dimensions = 2):

    # Dataset
    folder = baseDir + nome + '/'
    # Creazione hmm
    topology_factory = HiddenMarkovModelTopology()
    emissions = create_gesture_emissions(n_states, scale=100)
    model = topology_factory.forward(nome, n_states, emissions)
    # training leave one out
    model = training_leave_one_out(model, folder, index, dimensions=dimensions)
    return model


    # Primitive
def create_primitive_model(baseDir, n_states, direction, degree = 0):
    # Left, Right, Up, Down
    if(direction == Direction.left):
        dataset = LeapDataset(baseDir + 'down-trajectory/left/')
    elif(direction == Direction.right):
        dataset = LeapDataset(baseDir + 'down-trajectory/right/')
    elif(direction == Direction.up):
        dataset = LeapDataset(baseDir + 'down-trajectory/up/')
    elif(direction == Direction.down):
        dataset = LeapDataset(baseDir + 'down-trajectory/down/')
    # Diagonal
    elif (direction == Direction.diagonal):
        dataset = LeapDataset(baseDir + 'down-trajectory/diagonal_{}/'.format(degree))
    # Forward Behind
    elif direction == Direction.forward or direction == Direction.behind:
        if direction == Direction.forward:
            dataset = LeapDataset(baseDir + 'down-trajectory/forward/')
        else:
            dataset = LeapDataset(baseDir + 'down-trajectory/behind/')

        model = create_swipe_model(direction, n_states)
        model.fit(dataset.read_dataset(3, 100), use_pseudocount=True)
        return model

    model = create_swipe_model(direction, n_states)
    model.fit(dataset.read_dataset(2, 100), use_pseudocount=True)

    return model

#### Dataset Gesture e Primitive ####
def create_primitive_dataset(baseDir, direction, degree=0, sample=20):

    if direction == Direction.left:
        dataset = LeapDataset(baseDir + 'original/left/')
        nome = 'left'
    elif direction == Direction.right:
        dataset = LeapDataset(baseDir + 'original/right/')
        nome = 'right'
    # Creazione cartella che ospiterà i file
    elif direction == Direction.up:
        dataset = LeapDataset(baseDir + 'original/right/')
        nome = 'up'
    elif direction == Direction.down:
        dataset = LeapDataset(baseDir + 'original/left/')
        nome = 'down'
    elif direction == Direction.diagonal:
        dataset = LeapDataset(baseDir + 'original/right/')
        nome = 'diagonal_{}'.format(degree)
    elif direction == Direction.forward :
        dataset = LeapDataset(baseDir + 'original/right/')
        nome = 'forward'
    elif direction == Direction.behind:
        dataset = LeapDataset(baseDir + 'original/left/')
        nome = 'behind'

    #create_folder(baseDir, nome)

    # Creazione dataset
    # Up or Down
    if direction == Direction.up or direction == Direction.down:
        # Swap
        dataset.swap(baseDir + 'original/' + nome + '/', nome)
    # Diagonal
    elif direction == Direction.diagonal:
        # Rotate
        dataset.rotate_lines(baseDir + 'original/' + nome + '/', nome, degree)
    # Forward or Behind
    elif direction == Direction.forward or direction == Direction.behind:
        # Swap z
        dataset.swap(baseDir + 'original/' + nome + '/', nome, dimensions = 3)

    dataset = LeapDataset(baseDir + 'original/' + nome + '/')
    # Normalizzazione
    dataset.normalise(baseDir + 'normalised-trajectory/' + nome + '/', norm_axis=False)
    dataset = LeapDataset(baseDir + 'normalised-trajectory/' + nome + '/')
    # DownSample
    dataset.down_sample(baseDir + 'down-trajectory/' + nome + '/', sample)
    dataset = LeapDataset(baseDir + 'down-trajectory/' + nome + '/')

    # Plot
    for filename in dataset.getCsvDataset():
        sequence = dataset.read_file(filename, dimensions=2, scale=100)
    plt.axis("equal")
    plt.plot(sequence[:, 0], sequence[:, 1])
    plt.show()

def create_gesture_dataset(baseDir, path, sample=20):
    # Creazione cartelle
    create_folder(baseDir, path)
    # Creazione dataset originale
    LeapDataset.replace_csv(baseDir + 'original/' + path)
    dataset = LeapDataset(baseDir + 'original/' + path)
    # Normalizzazione
    dataset.normalise(baseDir + 'normalised-trajectory/' + path, norm_axis=True)
    dataset = LeapDataset(baseDir + 'normalised-trajectory/' + path)
    # DownSample
    dataset.down_sample(baseDir + 'down-trajectory/' + path, sample)

    dataset = LeapDataset(baseDir + 'down-trajectory/' + path + '/')
    # Plot
    for filename in dataset.getCsvDataset():
        sequence = dataset.read_file(filename, dimensions=2, scale=100)
        plt.axis("equal")
        plt.plot(sequence[:, 0], sequence[:, 1])
        plt.show()

def create_folder(baseDir, nome, operator = '', is_ground = True):
    if(is_ground == True):
        # Creazione cartelle
        if not os.path.exists(baseDir + 'original/' + nome):
            os.makedirs(baseDir + 'original/' + nome)
        if not os.path.exists(baseDir + 'normalised-trajectory/' + nome):
            os.makedirs(baseDir + 'normalised-trajectory/' + nome)
        if not os.path.exists(baseDir + 'down-trajectory/' + nome):
            os.makedirs(baseDir + 'down-trajectory/' + nome)
    else:
        # Creazione cartelle per gesture composte
        if not os.path.exists(baseDir + operator+'/'+nome):
            os.makedirs(baseDir + operator+'/'+nome)
