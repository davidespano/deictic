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


def create_swipe_emissions(n_states, direction= Direction.left, scale=1):
    distributions = []
    step = scale / n_states;
    for i in range(0, n_states):
        if direction == Direction.left:
            a = scale - (i * step);
        elif direction == Direction.right:
            a = i * step
        b = random.random() * scale
        gaussianX = NormalDistribution(a, scale * 0.1)
        gaussianY = NormalDistribution(scale * 0.1, scale * 0.1)
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
    swipe = topology_factory.forward(name, n_states, emissions)
    return swipe


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
    sequences = leftDataset.parallel_merge(leftDataset, dimensions=2, scale=1, dis=True)
    for i in range(0, len(sequences)):
        numpy.savetxt(baseDir + 'parallel/left-left-parallel/left-left-{}.csv'.format(i), sequences[i], delimiter=',')

    # Prove
    print('Sinsitra')
    wrong_test(parallel, baseDir + 'parallel/left-left-parallel/', dimensions=4)
    #print('Sinistra + Destra non sfasate')
    #wrong_test_parallel(parallel, baseDir + 'parallel/left-right-swipe/', dimensions=4)
    #print('Sinistra + Destra sfasate')
    #wrong_test(parallel, baseDir + 'parallel/left-right-dis-swipe/')

    print(parallel.log_probability(parallel.sample()))

    print('fine')



baseDir = '/Users/davide/Google Drive/Dottorato/Database/Leap/csv/'

#test_1(baseDir, Direction.left, 4)
#test_1(baseDir, Direction.right, 4)

#test_sequence(baseDir, 4)
#test_choice(baseDir, 4)
#test_iterative(baseDir, 4)

#test_disabling(baseDir, 4)

test_parallel(baseDir, 4)





