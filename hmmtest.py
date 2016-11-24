from dataset import *
from pomegranate import *
from topology import *
import matplotlib.pyplot as plt
import random
from enum import Enum

random.seed(0)


class Direction(Enum):
    left = 3
    right = 1
    up = 2
    down = 4
    diagonal_up = 5
    diagonal_down = 6

class TypeTest(Enum):
    rectangle = 1
    triangle = 2


#### Creazione HMM ####
def create_swipe_emissions(n_states, direction= Direction.left, scale=1):
    distributions = []
    step = scale / n_states;
    for i in range(0, n_states):
        if direction == Direction.left:
            a = scale - (i * step)
            b = random.random() * scale
        elif direction == Direction.right:
            a = i * step
            b = random.random() * scale
        elif direction == Direction.up:
            a = random.random() * scale
            b = i * step
        elif direction == Direction.down:
            a = random.random() * scale
            b = scale - (i * step)
        elif direction == Direction.diagonal_up:
            a = i * step
            b = i * step
        elif direction == Direction.diagonal_down:
            a = scale - (i * step)
            b = scale - (i * step)

        gaussianX = NormalDistribution(a, scale * 0.1)
        gaussianY = NormalDistribution(b, scale * 0.1)
        distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY]))
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

def create_swipe_model(direction, n_states, name=None):
    # create the hmm model
    topology_factory = HiddenMarkovModelTopology()
    emissions = create_swipe_emissions(n_states, direction, scale=100)
    if name is None:
        if direction == Direction.left:
            name = 'left-swipe'
        elif direction == Direction.right:
            name = 'right-swipe'
        elif direction == Direction.up:
            name = 'up-swipe'
        elif direction == Direction.down:
            name = 'down-swipe'
        elif direction == Direction.diagonal_up:
            name = 'diagonal_up-swipe'
        elif direction == Direction.diagonal_down:
            name = 'diagonal_down-swipe'

    swipe = topology_factory.forward(name, n_states, emissions)
    return swipe

def create_complete_model(n_states, name=None):
    # Creazione hmm
    topology_factory = HiddenMarkovModelTopology()
    emissions = create_swipe_emissions(n_states, scale=100)
    swipe = topology_factory.forward(name, n_states, emissions)
    return swipe

#### Training & Valutazione ####
def leave_one_out(model, correct_dir, index):
    # load dataset with correct examples and apply LOO technique
    correct = LeapDataset(correct_dir)
    one, sequences = correct.leave_one_out(index, dimensions=2, scale = 100)


    # train the hmm
    model.fit(sequences, use_pseudocount=True)

    print('Leave one out log-probability: {}'.format(model.log_probability(one)))

    return model


def wrong_test(model, wrong_dir, dimensions=2):
    wrong = LeapDataset(wrong_dir)

    for filename in wrong.getCsvDataset():
        sequence = wrong.read_file(filename, dimensions, scale=100)
        print('{} log-probability: {}, normalised-log-probability {}'.format(
            filename, model.log_probability(sequence),
            model.log_probability(sequence) / len(sequence)
        ))
    plt.plot(sequence)
    plt.show()

def plot_sample(model, samples = 3):
    fig = plt.figure();
    plt.axis('equal')
    for i in range(0, samples):
        sample = model.sample()
        print(model.log_probability(sample))
        seq = numpy.array(sample).astype('float')
        plt.plot(seq[:, 0], seq[:, 1], marker='.')
    plt.show()

def viterbi_seq(model, dir):
    data = LeapDataset(dir)
    for filename in data.getCsvDataset():
        sequence = data.read_file(filename, dimensions=2, scale=100)
        logp, states = model.viterbi(sequence)
        print('{} log-probability: {}'.format(filename, logp))
        print('{} state sequence: {}'.format(filename, ' '.join(state.name for idx, state in states[1:])))

#### Test ####
def test_1(baseDir, direction, n_states):
    if direction == Direction.left:
        correct = 'down-trajectory/left/'
        wrong = 'down-trajectory/right/'
    elif direction == Direction.right:
        correct = 'down-trajectory/right/'
        wrong = 'down-trajectory/left/'

    base = create_swipe_model(direction, n_states)

    base.plot()
    plt.show()

    for i in range(0, 13):
        print('**************  model {} *****************'.format(i))
        model = leave_one_out(base.copy(), baseDir + correct, i)
        print('--------------------------------------------')
        wrong_test(model, baseDir + wrong)
        print()
        print()

def test_sequence(baseDir, n_states):
    right = create_swipe_model(Direction.right, n_states)
    left = create_swipe_model(Direction.left, n_states)

    leftDataset = LeapDataset(baseDir + 'down-trajectory/left/')
    rightDataset = LeapDataset(baseDir + 'down-trajectory/right/')

    # train the hmm
    left.fit(leftDataset.read_dataset(2, 100), use_pseudocount=True)
    right.fit(rightDataset.read_dataset(2, 100), use_pseudocount=True)

    left, seq_edges = HiddenMarkovModelTopology.sequence([left, right]);

    print(seq_edges)

    left.plot()
    plt.show()

    #print('Sequenza dati sinistra')
    #viterbi_seq(left, baseDir + 'down-trajectory/left/')

    #print('Sequenza dati destra')
    #viterbi_seq(left, baseDir + 'down-trajectory/right/')

    print('Solo destra')
    wrong_test(left, baseDir + 'down-trajectory/right/')
    print()


    #sequences = leftDataset.sequence_merge(rightDataset, dimensions=2)
    #for i in range(0, len(sequences)):
    #    numpy.savetxt(baseDir + 'sequence/left-right-swipe{}.csv'.format(i), sequences[i], delimiter=',')
    wrong_test(left, baseDir + 'sequence/')

def test_choice(baseDir, n_states):
    right = create_swipe_model(Direction.right, n_states)
    left = create_swipe_model(Direction.left, n_states)

    leftDataset = LeapDataset(baseDir + 'down-trajectory/left/')
    rightDataset = LeapDataset(baseDir + 'down-trajectory/right/')

    # train the hmm
    left.fit(leftDataset.read_dataset(2, 100), use_pseudocount=True)
    right.fit(rightDataset.read_dataset(2, 100), use_pseudocount=True)

    left, seq_edges = HiddenMarkovModelTopology.choice([left, right])

    left.plot()
    plt.show()

    print('Solo destra')
    wrong_test(left, baseDir + 'down-trajectory/right/')
    print()

    print('Solo sinistra')
    wrong_test(left, baseDir + 'down-trajectory/left/')
    print()

    print('Sequence')
    wrong_test(left, baseDir + 'sequence/')
    viterbi_seq(left, baseDir + 'sequence/')
    print()

        #left.add_transition(left.start, right.states[right.start_index])

def test_iterative(baseDir, n_states):
    left = create_swipe_model(Direction.left, n_states)

    leftDataset = LeapDataset(baseDir + 'down-trajectory/left/')

    # train the hmm
    left.fit(leftDataset.read_dataset(2, 100), use_pseudocount=True)

    sequences = leftDataset.sequence_merge(leftDataset, dimensions=2)
    for i in range(0, len(sequences)):
        numpy.savetxt(baseDir + 'iterative/left/left-left-swipe{}.csv'.format(i), sequences[i], delimiter=',')

    left, seq_edges = HiddenMarkovModelTopology.iterative(left)

    print(seq_edges)

    print('Solo destra')
    wrong_test(left, baseDir + 'down-trajectory/right/')
    print()

    print('Solo sinistra')
    wrong_test(left, baseDir + 'down-trajectory/left/')
    print()

    print('Due left')
    wrong_test(left, baseDir + 'iterative/left/')
    print()

def test_disabling(baseDir, n_states):

    left1 = create_swipe_model(Direction.left, n_states)
    left2 = create_swipe_model(Direction.left, n_states, "dis-term")
    right = create_swipe_model(Direction.right, n_states)

    #disabling1, seq_edges = HiddenMarkovModelTopology.disabling(left1.copy(), right.copy())

    #it_left1, seq_edges = HiddenMarkovModelTopology.iterative(left1)
    #disabling2, seq_edges = HiddenMarkovModelTopology.disabling(it_left1, right.copy())

    seq_lr, seq_edges = HiddenMarkovModelTopology.sequence([left1, right])
    #disabling3, seq_edges = HiddenMarkovModelTopology.disabling(seq_lr.copy(), left2.copy(), seq_edges)

    it_seq, seq_edges = HiddenMarkovModelTopology.iterative(seq_lr, seq_edges)
    disabling4, seq_edges = HiddenMarkovModelTopology.disabling(it_seq, left2, seq_edges)

    #it_seq.plot()
    #plt.show()

    for edge in seq_edges:
        print('edge ({}, {})'.format(edge[0].name, edge[1].name))

def test_parallel(baseDir, n_states):
    left = create_swipe_model(Direction.left, n_states)
    right = create_swipe_model(Direction.right, n_states)
    # Training
    leftDataset = LeapDataset(baseDir + 'down-trajectory/left/')
    rightDataset = LeapDataset(baseDir + 'down-trajectory/right/')
    left.fit(leftDataset.read_dataset(2, 100), use_pseudocount=True)
    right.fit(rightDataset.read_dataset(2, 100), use_pseudocount=True)

    # definizione parallel
    parallel, seq_edges = HiddenMarkovModelTopology.parallel(left, right)

    #parallel.plot();
    #plt.show()

    # Creazione dataset
    #sequences = leftDataset.parallel_merge(rightDataset, dimensions=2, scale=1, flag=False)
    #for i in range(0, len(sequences)):
        #numpy.savetxt(baseDir + 'parallel/left-right-trasl-parallel/left-right-{}.csv'.format(i), sequences[i], delimiter=',')

    # Prove
    #print('Sinsitra')
    #wrong_test(parallel, baseDir + 'parallel/left-right-trasl-parallel/', dimensions=4)
    #print('Sinistra + Destra non sfasate')
    #wrong_test_parallel(parallel, baseDir + 'parallel/left-right-swipe/', dimensions=4)
    #print('Sinistra + Destra sfasate')
    #wrong_test(parallel, baseDir + 'parallel/left-right-dis-swipe/')
    #print(parallel.log_probability(parallel.sample()))

def test_leap(baseDir, n_states, typeTest):

    #### Test ####
    # Rectangle
    if(typeTest == TypeTest.rectangle):
        # Creazione hmm primitive
        left = create_swipe_model(Direction.left, n_states)
        right = create_swipe_model(Direction.right, n_states)
        up = create_swipe_model(Direction.up, n_states)
        down = create_swipe_model(Direction.down, n_states)
        # Training hmm primitive
        leftDataset = LeapDataset(baseDir + 'down-trajectory/left/')
        rightDataset = LeapDataset(baseDir + 'down-trajectory/right/')
        upDataset = LeapDataset(baseDir + 'down-trajectory/up/')
        downDataset = LeapDataset(baseDir + 'down-trajectory/down/')
        left.fit(leftDataset.read_dataset(2, 100), use_pseudocount=True)
        right.fit(rightDataset.read_dataset(2, 100), use_pseudocount=True)
        up.fit(upDataset.read_dataset(2, 100), use_pseudocount=True)
        down.fit(downDataset.read_dataset(2, 100), use_pseudocount=True)

        # Creazione hmm gesture completa
        rectangle_model, seq_edges = HiddenMarkovModelTopology.sequence([down, right, up, left])
        # Prova
        wrong_test(rectangle_model, baseDir + 'down-trajectory/rectangle/')
        print()
        wrong_test(rectangle_model, baseDir + 'down-trajectory/triangle/')
    # Triangle
    if(typeTest == TypeTest.triangle):
        # Creazione hmm Primitive
        right = create_swipe_model(Direction.right, n_states)
        up_45 = create_swipe_model(Direction.diagonal_up, n_states)
        down_m_45 = create_swipe_model(Direction.diagonal_down, n_states)
        # Training hmm primitive
        rightDataset = LeapDataset(baseDir + 'down-trajectory/right/')
        up_45Dataset = LeapDataset(baseDir + 'down-trajectory/diagonal_up_45/')
        down_m_45Dataset = LeapDataset(baseDir + 'down-trajectory/diagonal_down_-45/')
        right.fit(rightDataset.read_dataset(2, 100), use_pseudocount=True)
        up_45.fit(up_45Dataset.read_dataset(2, 100), use_pseudocount=True)
        down_m_45.fit(down_m_45Dataset.read_dataset(2, 100), use_pseudocount=True)
        # Creazione hmm gesture completa
        triangle_model, seq_edges = HiddenMarkovModelTopology.sequence([down_m_45, right, up_45])

        wrong_test(up_45, baseDir + 'down-trajectory/left/')
        print()
        wrong_test(up_45, baseDir + 'down-trajectory/right/')
        print()
        wrong_test(up_45, baseDir + 'down-trajectory/up/')
        print()
        wrong_test(up_45, baseDir + 'down-trajectory/down/')
        # Prova
        #wrong_test(triangle_model, baseDir + 'down-trajectory/triangle/')
        print()
        #wrong_test(triangle_model, baseDir + 'down-trajectory/rectangle/')

def create_primitive_dataset(baseDir, direction, degree = 0, sample = 20):

    # Creazione cartella che ospiterà i file
    if direction == Direction.up:
        dataset = LeapDataset(baseDir + 'original/right/')
        nome = 'up'
    elif direction == Direction.down:
        dataset = LeapDataset(baseDir + 'original/left/')
        nome = 'down'
    elif direction == Direction.diagonal_up:
        dataset = LeapDataset(baseDir + 'original/right/')
        nome = 'diagonal_up_{}'.format(degree)
    elif direction == Direction.diagonal_down:
        dataset = LeapDataset(baseDir + 'original/left/')
        nome = 'diagonal_down_{}'.format(degree)

    create_folder(baseDir, nome)

    # Creazione dataset
    # Up or Down
    if direction == Direction.up or direction == Direction.down:
        # Swap
        dataset.swap(baseDir + 'original/' + nome + '/', nome)
    # Diagonal
    elif direction == Direction.diagonal_up or direction == Direction.diagonal_down:
        # Crea file originali
        dataset.rotate_lines(baseDir + 'original/' + nome + '/', nome, degree)

    dataset = LeapDataset(baseDir + 'original/' + nome + '/')
    # Normalizzazione
    #dataset.normalise(baseDir + 'normalised-trajectory/' + nome + '/', norm_axis=True)
    #dataset = LeapDataset(baseDir + 'normalised-trajectory/' + nome + '/')
    # DownSample
    #dataset.down_sample(baseDir + 'down-trajectory/' + nome + '/', sample)
    #dataset = LeapDataset(baseDir + 'down-trajectory/'+nome+'/')

    # Plot
    for filename in dataset.getCsvDataset():
        sequence = dataset.read_file(filename, dimensions = 2, scale=100)
        plt.axis("equal")
        plt.plot(sequence[:,0], sequence[:,1])
        plt.show()

def create_gesture_dataset(baseDir, path, sample = 20):
    # Creazione cartelle
    create_folder(baseDir, path)
    # Replace
    LeapDataset.replace_csv('', baseDir + 'original/'+path)
    # Creazione dataset originale
    dataset = LeapDataset(baseDir+'original/'+path)
    # Normalizzazione
    dataset.normalise(baseDir+'normalised-trajectory/'+path, norm_axis = True)
    dataset = LeapDataset(baseDir+'normalised-trajectory/'+path)
    # DownSample
    dataset.down_sample(baseDir+'down-trajectory/'+path, sample)


def create_folder(baseDir, nome):
    # Creazione cartelle
    if not os.path.exists(baseDir + 'original/' + nome):
        os.makedirs(baseDir + 'original/' + nome)
    if not os.path.exists(baseDir + 'normalised-trajectory/' + nome):
        os.makedirs(baseDir + 'normalised-trajectory/' + nome)
    if not os.path.exists(baseDir + 'down-trajectory/' + nome):
        os.makedirs(baseDir + 'down-trajectory/' + nome)


#baseDir = '/Users/davide/Google Drive/Dottorato/Database/Leap/csv/'
baseDir = '/home/alessandro/Scaricati/csv/'
#test_1(baseDir, Direction.left, 4)
#test_1(baseDir, Direction.right, 4)

#### Test Primitive ####
#test_sequence(baseDir, 4)
#test_choice(baseDir, 4)
#test_iterative(baseDir, 4)
#test_disabling(baseDir, 4)
#test_parallel(baseDir, 4)

#### Test Gesture Complete ####
# Crea file csv originali gesture
#LeapDataset.find_gesture_file('', '/home/alessandro/Scaricati/gestures/', baseDir, 'rectangle')
#LeapDataset.find_gesture_file('', '/home/alessandro/Scaricati/gestures/', baseDir, 'triangle')

# Crea file csv primitive
# Up
#create_primitive_dataset(baseDir, Direction.up)
# Down
#create_primitive_dataset(baseDir, Direction.down)
# Diagonal Up
create_primitive_dataset(baseDir, Direction.diagonal_up, 45)
# Diagonal Down
#create_primitive_dataset(baseDir, Direction.diagonal_down, -45)

# Crea file csv gesture
# Rectangle
#create_gesture_dataset(baseDir, 'rectangle/', 72)
# Triangle
#create_gesture_dataset(baseDir, 'triangle/', 54)

# Test
#print('Rectangle')
#test_leap(baseDir, 4, TypeTest.rectangle)
#print('Triangle')
#test_leap(baseDir, 4, TypeTest.triangle)


print('Fine')