from dataset import *
from pomegranate import *
from topology import *
import matplotlib.pyplot as plt
import random
from gesture import *
from test import *
from enum import Enum

#### Valutazione ####
def viterbi_seq(model, dir):
    data = LeapDataset(dir)
    for filename in data.getCsvDataset():
        sequence = data.read_file(filename, dimensions=2, scale=100)
        logp, states = model.viterbi(sequence)
        print('{} log-probability: {}'.format(filename, logp))
        print('{} state sequence: {}'.format(filename, ' '.join(state.name for idx, state in states[1:])))

def plot_sample(model, samples=3):
    fig = plt.figure();
    plt.axis('equal')
    for i in range(0, samples):
        sample = model.sample()
        print(model.log_probability(sample))
        seq = numpy.array(sample).astype('float')
        plt.axis('equal')
        plt.plot(seq[:, 0], seq[:, 1], marker='.')
        plt.show()

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
    left.plot()
    plt.show()


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

    disabling4.plot()
    plt.show()

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

    parallel.plot();
    plt.show()

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


### Dataset ###
def create_operators_dataset(baseDir, operator, dimensions=2, scale=1, iterazioni = 1):
    nome = []
    # Caret
    nome.append('caret')
    # Delete
    nome.append('delete')
    # Left
    nome.append('left')
    # Rectangle
    nome.append('rectangle')
    # Right
    nome.append('right')
    # Square Braket Left
    nome.append('square-braket-left')
    # Square Braket Right
    nome.append('square-braket-right')
    # Triangolo
    nome.append('triangle')
    # V
    nome.append('v')
    # X
    nome.append('x')

    gestures = []
    for index in range(0, len(nome), iterazioni):
        random.seed()

        # Choice
        if (operator == Operator.choice):
            operatore = 'choice'
            #sequence = first.choice_

        # Disabling : iterative + ground
        elif (operator == Operator.disabling):
            num_rand_1 = index
            num_rand_2 = index + 1

            first = LeapDataset(baseDir + 'down-trajectory/' + nome[num_rand_1] + '/')
            second = LeapDataset(baseDir + 'down-trajectory/' + nome[num_rand_2] + '/')
            filename = nome[num_rand_1] +'_'+ nome[num_rand_2]
            # Crea sequenze
            operatore = 'disabling'
            sequences = first.disabling_merge(first, second, dimensions=dimensions, scale=scale)

        # Iterative
        elif (operator == Operator.iterative):
            first = LeapDataset(baseDir + 'down-trajectory/' + nome[index] + '/')
            filename = nome[index]

            # Crea sequenze
            operatore = 'iterative'
            sequences = first.iterative_merge(iterazioni=2, dimensions=2, scale=1)

        # Parallel
        elif (operator == Operator.parallel):
            num_rand_1 = index
            num_rand_2 = index + 1

            first = LeapDataset(baseDir + 'down-trajectory/' + nome[num_rand_1] + '/')
            second = LeapDataset(baseDir + 'down-trajectory/' + nome[num_rand_2] + '/')
            filename = nome[num_rand_1] +'_'+ nome[num_rand_2]

            operatore = 'parallel'
            sequences = first.parallel_merge(second, dimensions=dimensions, scale=scale, flag_trasl=False)

        # Sequence
        elif (operator == Operator.sequence):
            num_rand_1 = index
            num_rand_2 = index+1
            first = LeapDataset(baseDir + 'down-trajectory/' + nome[num_rand_1] + '/')
            second = LeapDataset(baseDir + 'down-trajectory/' + nome[num_rand_2] + '/')
            filename = nome[num_rand_1] + '_' + nome[num_rand_2]

            operatore = 'sequence'
            sequences = first.sequence_merge([second], dimensions=dimensions, scale=scale)

        gestures.append(filename)
        # Salva file
        create_folder(baseDir, filename, operator = operatore, is_ground = False)
        for i in range(0, len(sequences)):
            numpy.savetxt(baseDir + operatore + '/' + filename + '/' + filename + '_{}.csv'.format(i), sequences[i],
                          delimiter=',')


### Prove ###
def test_primitive(baseDir, n_states, direction, degree = 0):
    dimensions = 2

    # Forward
    if direction == Direction.forward:
        model = create_3d_swipe_model(direction = Direction.forward, n_states = n_states)
        dimensions = 3
    # Behind
    elif direction == Direction.behind:
        model = create_3d_swipe_model(direction=Direction.behind, n_states = n_states)
        dimensions = 3
    # Left, Right, Up, Down, Diagonal
    else:
        model = create_swipe_model(direction=direction, n_states=n_states)

    print('Left')
    wrong_test(model, baseDir + 'down-trajectory/left/', dimensions= dimensions)
    print('Right')
    wrong_test(model, baseDir + 'down-trajectory/right/', dimensions=dimensions)
    print('Up')
    wrong_test(model, baseDir + 'down-trajectory/up/', dimensions=dimensions)
    print('Down')
    wrong_test(model, baseDir + 'down-trajectory/down/', dimensions= dimensions)
    print('Diagonal')
    wrong_test(model, baseDir+'down-trajectory/diagonal_{}/'.format(degree), dimensions=dimensions)
    print('Forward')
    wrong_test(model, baseDir + 'down-trajectory/forward/', dimensions= dimensions)
    print('Behind')
    wrong_test(model, baseDir + 'down-trajectory/behind/', dimensions=dimensions)

    # Plotta sample
    plot_sample(model, 3)

def test_leap(baseDir, n_states, typeTest):

    #### Test ####
    # Rectangle
    if(typeTest == TypeTest.rectangle):
        model = create_rectangle(baseDir, n_states)

    # Triangle
    elif(typeTest == TypeTest.triangle):
        model = create_triangle(baseDir, n_states)

    # Caret
    elif typeTest == TypeTest.caret:
        model = create_caret(baseDir, n_states)

    # V
    elif typeTest == TypeTest.v:
        model = create_v(baseDir, n_states)

    # X
    elif typeTest == typeTest.x:
        model = create_x(baseDir, n_states)

    # Square Bracket Left
    elif typeTest == typeTest.square_braket_left:
        model = create_square_braket_left(baseDir, n_states)

    # Square Bracket Right
    elif typeTest == typeTest.square_braket_right:
        model = create_square_braket_right(baseDir, n_states)

    # Delete
    elif typeTest == typeTest.delete:
        model = create_delete(baseDir, n_states)

    # Star
    elif typeTest == typeTest.star:
        model = create_star(baseDir, n_states)

    # Plotta sample
    plot_sample(model, 3)

    # Prova
    print('Caret')
    wrong_test(model, baseDir + 'down-trajectory/caret/')
    print()
    print('Rectangle')
    wrong_test(model, baseDir + 'down-trajectory/rectangle/')
    print()
    print('Triangle')
    wrong_test(model, baseDir + 'down-trajectory/triangle/')
    print()
    print('V')
    wrong_test(model, baseDir + 'down-trajectory/v/')
    print()
    print('X')
    wrong_test(model, baseDir + 'down-trajectory/x/')
    print('Square Bracket Left')
    wrong_test(model, baseDir + 'down-trajectory/square-braket-left/')
    print('Square Bracket Right')
    wrong_test(model, baseDir + 'down-trajectory/square-braket-right/')
    print('Delete')
    wrong_test(model, baseDir + 'down-trajectory/delete/')
    print('Star')
    wrong_test(model, baseDir + 'down-trajectory/star/')

def compare_models(baseDir, n_states):
    #### Test ####
    models = []
    # Caret
    model, seq = create_caret(baseDir, n_states)
    models.append(model)
    # Delete
    model, seq = create_delete(baseDir, n_states)
    models.append(model)
    # Left
    model = create_primitive_model(baseDir, n_states, direction = Direction.left)
    models.append(model)
    # Rectangle
    model, seq = create_rectangle(baseDir, n_states)
    models.append(model)
    # Right
    model = create_primitive_model(baseDir, n_states, direction = Direction.right)
    models.append(model)
    # Square Braket Left
    model, seq = create_square_braket_left(baseDir, n_states)
    models.append(model)
    # Square Braket Right
    model, seq = create_square_braket_right(baseDir, n_states)
    models.append(model)
    # Triangle
    model, seq = create_triangle(baseDir, n_states)
    models.append(model)
    # V
    model, seq = create_v(baseDir, n_states)
    models.append(model)
    # X
    model, seq = create_x(baseDir, n_states)
    models.append(model)

    compare_all_models_test(models, baseDir+'down-trajectory/')

    #sequence = triangle.sample()
    #result = numpy.array(sequence).astype('float')
    #plt.axis("equal")
    #plt.plot(result[:, 0], result[:, 1])
    #plt.show()

def compare_composite_models(baseDir, n_states, operator, dimensions = 2):
    # Determina operazione
    models_complete = []
    # Choice
    if (operator == Operator.choice):
        operatore = 'choice/'
    # Disabling : iterative + ground
    elif (operator == Operator.disabling):
        # Crea sequenze
        operatore = 'disabling/'
    # Iterative
    elif (operator == Operator.iterative):
        # Crea sequenze
        operatore = 'iterative/'
    # Parallel
    elif (operator == Operator.parallel):
        operatore = 'parallel/'
    # Sequence
    elif (operator == Operator.sequence):
        operatore = 'sequence/'

    # Scorri file nella cartella
    folders = LeapDataset.get_immediate_subdirectories(baseDir+'/'+operatore)
    folders = sorted(folders)  # Riordina cartelle
    for folder in folders:
        names = folder.split('_')
        models = []
        for name in names:
            if name == 'caret':
                type = TypeTest.caret
            elif name == 'delete':
                type = TypeTest.delete
            elif name == 'left':
                type = TypeTest.left_swipe
            elif name == 'rectangle':
                type = TypeTest.rectangle
            elif name == 'square-braket-left':
                type = TypeTest.square_braket_left
            elif name == 'square-braket-right':
                type = TypeTest.square_braket_right
            elif name == 'triangle':
                type = TypeTest.triangle
            elif name == 'v':
                type = TypeTest.v
            elif name == 'x':
                type = TypeTest.x
            elif name == 'right':
                type = TypeTest.right_swipe

            model, seq = create_gesture(type, baseDir, n_states)
            models.append(model)

        # Crea modello
        if (operator == Operator.choice):
            complete_model, seq = create_choice(models)
            models_complete.append(complete_model)
        # Disabling : iterative + ground
        elif (operator == Operator.disabling):
            iterative, seq = create_iterative(models[0])
            complete_model, seq = create_disabling(iterative, models[1], seq)
            models_complete.append(complete_model)
        # Iterative
        elif (operator == Operator.iterative):
            complete_model, seq = create_iterative(models[0])
            models_complete.append(complete_model)
        # Parallel
        elif (operator == Operator.parallel):
            # Cambia nomi stati modello
            for model in models:
                for indice in range(0, len(model.states)):
                    state = model.states[indice]
                    state.name = model.name+'_'+state.name+'_{}'.format(indice)
            complete_model, seq = create_parallel(models[0], models[1])
            models_complete.append(complete_model)
        # Sequence
        elif (operator == Operator.sequence):
            complete_model, seq = create_sequence(models)
            models_complete.append(complete_model)

    compare_all_models_test(models_complete, baseDir+'/'+operatore, dimensions = dimensions)

def compare_models_hmm_gesture(baseDir, n_states, index = 1):
    baseDir = baseDir + 'down-trajectory/'
    #### Test ####
    models = []
    models.append(create_hmm_gesture_complete('caret', baseDir, n_states*2, index))
    models.append(create_hmm_gesture_complete('delete', baseDir, n_states*3, index))
    models.append(create_hmm_gesture_complete('left', baseDir, n_states, index))
    models.append(create_hmm_gesture_complete('rectangle', baseDir, n_states*4, index))
    models.append(create_hmm_gesture_complete('right', baseDir, n_states, index))
    models.append(create_hmm_gesture_complete('square-braket-left', baseDir, n_states*3, index))
    models.append(create_hmm_gesture_complete('square-braket-right', baseDir, n_states*3, index))
    models.append(create_hmm_gesture_complete('triangle', baseDir, n_states*3, index))
    models.append(create_hmm_gesture_complete('v', baseDir, n_states*2, index))
    models.append(create_hmm_gesture_complete('x', baseDir, n_states*3, index))

    compare_all_models_test_without_primitive(models, baseDir+'down-trajectory/', index = index)

def compare_composite_hmm_gesture(baseDir, operator, n_states, index = 1, dimensions = 2):
    # Determina operazione
    models_complete = []
    # Choice
    if (operator == Operator.choice):
        operatore = 'choice/'
    # Disabling : iterative + ground
    elif (operator == Operator.disabling):
        # Crea sequenze
        operatore = 'disabling/'
        n_states = n_states
    # Iterative
    elif (operator == Operator.iterative):
        # Crea sequenze
        operatore = 'iterative/'
        n_states = n_states
    # Parallel
    elif (operator == Operator.parallel):
        operatore = 'parallel/'
        n_states = n_states
    # Sequence
    elif (operator == Operator.sequence):
        operatore = 'sequence/'
        n_states = n_states

    baseDir = baseDir+'/'+operatore

    # Prendi cartelle e crea gesture
    folders = LeapDataset.get_immediate_subdirectories(baseDir)
    folders = sorted(folders)  # Riordina cartelle
    for folder in folders:
        models_complete.append(create_hmm_gesture_complete(folder, baseDir, n_states, index, dimensions = dimensions))

    # Compara tutti i modelli
    compare_all_models_test_without_primitive(models_complete, baseDir+'/', dimensions = dimensions, index = index)

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

# Crea file csv primitive
# Left
#create_primitive_dataset(baseDir, Direction.left)
# Right
#create_primitive_dataset(baseDir, Direction.right)
# Up
#create_primitive_dataset(baseDir, Direction.up)
# Down
#create_primitive_dataset(baseDir, Direction.down)
# Diagonal
#create_primitive_dataset(baseDir, Direction.diagonal, 45)
# Forward
#create_primitive_dataset(baseDir, Direction.forward)
# Behind
#create_primitive_dataset(baseDir, Direction.behind)

# Crea file csv originali gesture
#LeapDataset.find_gesture_file('', '/home/alessandro/Scaricati/gestures/', baseDir, 'rectangle')
#LeapDataset.find_gesture_file('', '/home/alessandro/Scaricati/gestures/', baseDir, 'triangle')
#LeapDataset.find_gesture_file('', '/home/alessandro/Scaricati/gestures/', baseDir, 'caret')
#LeapDataset.find_gesture_file('/home/alessandro/Scaricati/gestures/', baseDir, 'v')
#LeapDataset.find_gesture_file('/home/alessandro/Scaricati/gestures/', baseDir, 'x')
#LeapDataset.find_gesture_file('/home/alessandro/Scaricati/gestures/', baseDir, 'square-braket-left')
#LeapDataset.find_gesture_file('/home/alessandro/Scaricati/gestures/', baseDir, 'square-braket-right')
#LeapDataset.find_gesture_file('/home/alessandro/Scaricati/gestures/', baseDir, 'delete')
#LeapDataset.find_gesture_file('/home/alessandro/Scaricati/gestures/', baseDir, 'star')

# Crea file csv gesture
# Rectangle
#create_gesture_dataset(baseDir, 'rectangle/', 72)
# Triangle
#create_gesture_dataset(baseDir, 'triangle/', 54)
# Caret
#create_gesture_dataset(baseDir, 'caret/', 36)
# V
#create_gesture_dataset(baseDir, 'v/', 36)
# X
#create_gesture_dataset(baseDir, 'x/', 54)
# Square Bracket Left
#create_gesture_dataset(baseDir, 'square-braket-left/', 54)
# Square Bracket Right
#create_gesture_dataset(baseDir, 'square-braket-right/', 54)
# Delete
#create_gesture_dataset(baseDir, 'delete/', 54)
# Star
#create_gesture_dataset(baseDir, 'star/', 90)

# Crea dataset gesture composte
#create_operators_dataset(baseDir, Operator.iterative, dimensions=2, scale=1)

# Test - con primitive
#print('Rectangle')
#test_leap(baseDir, 8, TypeTest.rectangle) # Ok
#print('Triangle')
#test_leap(baseDir, 8, TypeTest.triangle) # Ok
#print('Caret')
#test_leap(baseDir, 8, TypeTest.caret) # Ok
#print('X')
#test_leap(baseDir, 8, TypeTest.x) # Ok
#print('Square braket left')
#test_leap(baseDir, 8, TypeTest.square_braket_left) # Ok
#print('Square braket left')
#test_leap(baseDir, 8, TypeTest.square_braket_left) # Ok
#print('Square braket right')
#test_leap(baseDir, 8, TypeTest.square_braket_right)# Ok
#print('Delete')
#test_leap(baseDir, 8, TypeTest.delete) # Ok anche se con square braket, x e rettangolo le differenze non sono nettissime (in ogni caso sono il doppio)
#print('Star')
#test_leap(baseDir, 8, TypeTest.star) # Ok
# Test - senza primitive

#print('V')
#test_leap(baseDir, 8, TypeTest.v) # Si confonde un poco con il rettangolo!

# Crea modelli
#compare_models(baseDir, 8) # Con primitive
#for i in range(13, 14): # Senza primitive
    #compare_models_hmm_gesture(baseDir, 8, index = i)
#compare_composite_models(baseDir, 8, Operator.parallel, dimensions = 4) # Composte con primitive
for i in range(0, 14): # Composte senza primitive
    compare_composite_hmm_gesture(baseDir, Operator.parallel, 8, index = i, dimensions = 4)


#model, seq = create_gesture(TypeTest.x, baseDir, 8)
#plot_sample(model, 3)

print('Fine')