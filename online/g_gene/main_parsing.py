#### Library ####
# Kalman filter
from pykalman import KalmanFilter
from dataset import *
from dataset.csvDataset import CsvDataset, CsvDatasetExtended
###
from model.gestureModel import TypeRecognizer
from gesture.datasetExpressions import DatasetExpressions
from gesture.createRecognizer import CreateRecognizer
from online.g_gene.model_factory import ModelFactory
from online.g_gene.trajectory_parsing import Parsing
# Test
from test.test import Test, ConfusionMatrix
# Levenshtein
from online.g_gene.levenshtein_distance import LevenshteinDistance
import itertools
####
from pomegranate import *

def plot_debug(directories=[], files=[]):
    # datasets
    dataset_kalman_resampled = {directory:transforms_set(directory).applyTransforms() for directory in directories}
    colors = ['r', 'b', 'g']

    for index in range(len(dataset_kalman_resampled[directories[0]])):
        data_plots = []
        # plotting
        k = 0
        plot_multiple = True if len(directories)>1 else False

        for directory in directories:
            item = dataset_kalman_resampled[directory][index]
            if not files or item[1] in files:
                data = item[0]
                label = Parsing.parsingLine(sequence=data).getLabelsSequence()
                data_plots.append(plt.plot(data[:, 0]+(k*10), data[:, 1], color=colors[k]))
                plt.scatter(data[:, 0]+(k*10), data[:, 1])
                for i in range(0, len(label)):
                    plt.annotate(str(label[i]), (data[i][0]+(k*10), data[i][1]))
                if plot_multiple == False:
                    plt.title(item[1])
                    plt.axis('equal')
                    plt.show()
        if plot_multiple == True:
            # Legend
            plt.title(directories)
            plt.axis('equal')
            plt.show()

def path_to_alignment(x, y, path):
    """
    This function will take in two sequences, and the ML path which is their alignment,
    and insert dashes appropriately to make them appear aligned. This consists only of
    adding a dash to the model sequence for every insert in the path appropriately, and
    a dash in the observed sequence for every delete in the path appropriately.
    """
    # convert A
    x = ''.join([a for a in x if a != 'A'])
    y = ''.join([a for a in y if a != 'A'])
    for i, (index, state) in enumerate(path[1:-1]):
        name = state.name

        if name.startswith('D'):
            y = y[:i] + '-' + y[i:]
        elif name.startswith('I'):
            x = x[:i] + '-' + x[i:]

    return x, y
### - ###

# base dir
base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/"
#  mode
debug = 4

if debug == 2:
    directories = ["arrow","caret"]#, "circle", "check", "delete_mark", "left_curly_brace", "left_sq_bracket",
                   #"pigtail", "question_mark", "rectangle", "right_curly_brace", "right_sq_bracket", "star", "triangle", "v", "x"]

    for directory in directories:
        print("Start " + directory)
        dataset = CsvDatasetExtended(base_dir + 'raw/' + directory + '/')
        # original kalman
        kalmanTransform = KalmanFilterTransform()
        dataset.addTransform(kalmanTransform)
        # resampling
        resampledTransform = ResampleTransform(delta=10)
        dataset.addTransform(resampledTransform)
        # parse
        parse = ParseSamples()
        dataset.addTransform(parse)
        # apply and save
        dataset.applyTransforms(base_dir + 'parsed/' + directory + '/')
        print("End " + directory)

if debug == 3:
    directories = ["arrow", "caret"]#, "circle", "check", "delete_mark", "left_curly_brace", "left_sq_bracket",
    #"pigtail", "question_mark", "rectangle", "right_curly_brace", "right_sq_bracket", "star", "triangle", "v", "x"]
    for directory in directories:
        print("Start " + directory)
        dataset = CsvDatasetExtended(base_dir + 'parsed/' + directory + '/', type=str)
        # read dataset #
        for file in dataset.readDataset():
            c = file.getPoints(columns=[0])
            print(c)
        print("End " + directory)

if debug == 4:
    ideal_sequences = {
        'arrow': [['O', 'A1', 'O', 'A5', 'O', 'A0', 'O', 'A6', 'O'],['O', 'A1', 'O', 'A4', 'O', 'A0', 'O', 'A6', 'O'],
                 ['O', 'A1', 'O', 'A4', 'O', 'A0', 'O', 'A5', 'O'],['O', 'A1', 'O', 'A5', 'O', 'A1', 'O'],
                 ['O', 'A1', 'O', 'A0', 'O', 'A6', 'O']],
        'caret': [['O', 'A2', 'O', 'A7', 'O'], ['O', 'A1', 'O', 'A6', 'O'], ['O', 'A1', 'O', 'A7', 'O'],
                  ['O', 'A2', 'O', 'A6', 'O']],
        # 'delete_mark': [['O', 'A7', 'O', 'A4', 'O', 'A1', 'O']],
        # 'left_sq_bracket': [['O', 'A4', 'O', 'A6', 'O', 'A0', 'O']],
        # 'rectangle': [['O', 'A6', 'O', 'A0', 'O', 'A2', 'O', 'A4', 'O']],
        # 'right_sq_bracket': [['O', 'A0', 'O', 'A6', 'O', 'A4', 'O']],
        # 'star': [['O', 'A1', 'O', 'A7', 'O', 'A3', 'O', 'A0', 'O', 'A5', 'O']],
        # 'triangle': [['O', 'A5', 'O', 'A0', 'O', 'A3', 'O']],
        # 'v': [['O', 'A7', 'O', 'A2', 'O'], ['O', 'A7', 'O', 'A1', 'O'], ['O', 'A6', 'O', 'A1', 'O'],
        #       ['O', 'A6', 'O', 'A2', 'O']],
        # 'x': [['O', 'A7', 'O', 'A2', 'O', 'A5', 'O']],
        # 'circle'            : [['O','A5','O','A7','O','A1','O','A3'],['O','BCCW','O','BCCW','O','BCCW','O','BCCW','O'],
        #                        ['O', 'A5', 'O', 'A6', 'O', 'A7', 'O', 'A1', 'O'],
        #                        ['O', 'A5', 'O', 'BCCW', 'O', 'A1', 'O', 'A3', 'O']],
        # 'left_curly_brace'  : [['O','BCCW','O','BCW','O','BCCW','O','BCW','O','BCCW','O'],
        #                        ['O','BCCW','O','BCCW','O','BCW','O','BCCW','O','BCW','O','BCCW','O']],
        # 'pigtail'           : [['O','A1','O','BCCW','O','A6','O'],['O','A1','O','BCCW','O'],
        #                        ['O', 'A0', 'O', 'A1', 'O', 'BCCW', 'O', 'BCCW', 'O']],
        # 'question_mark'     : [['O','BCW','O','BCCW','O'],['O','BCW','O','BCW','O','BCCW','O'],
        #                        ['O', 'A2', 'O', 'BCW', 'O', 'A5', 'O', 'A6', 'O'],
        #                        ['O', 'A1', 'O', 'A0', 'O', 'A6', 'O', 'A5', 'O']],
        # 'right_curly_brace' : [['O','BCW','O','BCCW','O','BCW','O','BCCW','O','BCW','O'],
        #                        ['O','BCW','O','BCW','O','BCCW','O','BCW','O','BCCW','O','BCW','O']]
    }

    models = {key:[ModelFactory.sequenceAlignment(name=key, ideal_sequence=item) for item in values] for key,values in ideal_sequences.items()}

    # test
    directories = [m for m in ideal_sequences]

    # confusion matrix
    confusion_matrix = ConfusionMatrix(directories)
    # neighbor
    best_hmm = 1
    for directory in directories:
        # original kalman + resampled + parsing
        dataset = CsvDatasetExtended(base_dir+'parsed/'+directory+'/', type=str)
        # apply transforms
        for sequence in dataset.readDataset():
            # get log_probabilities obtained from sequence
            points = [item for sublist in (sequence.getPoints(columns=[0])) for item in sublist]
            probabilities = Test.compare(points, models)
            # verify if row_label is contained in the firsts x elements (default, x=1)
            keys = []
            for key, value in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:best_hmm]:
                if isinstance(key, tuple):
                    keys = keys + list(itertools.chain(key))
                else:
                    keys.append(key)
            index_label = directory if directory in keys else keys[0]

            if directory not in keys:
                print('\n')
                print(points)
                print(probabilities)
                print(keys)

            # update confusion matrix
            confusion_matrix.update(row_label=directory, column_label=index_label,
                                                id_sequence=sequence.filename)

            #result.confusion_matrix.update(row_label=directory, column_label=column_label)

            # if column_label != directory:
            #     print("\n\nGesture recognized:{} ---- file:{}".format(column_label, sequence[1]))
            #     if column_label != directory:
            #         for name,hmms in models.items():
            #             for index in range(len(hmms)):
            #                 m=hmms[index]
            #                 logp, path = m.viterbi(sequence[0])
            #                 logp = (m.log_probability(sequence[0])/len(sequence[0]))
            #                 x, y = path_to_alignment(''.join(ideal_sequences[name][index]), ''.join(sequence[0]), path)
            #
            #                 print("Gesture:'{}' -- Described Sequence: {} -- Log_Probability: {}".format(''.join(m.name), ideal_sequences[m.name], logp))
            #                 print("--------Sequence            {}".format(sequence[0]))
            #                 print("--------Path:               {}".format([state.name for idx, state in path[1:-1]]))
            #                 print(x)
            #                 print(y)
    confusion_matrix.plot(normalize=False)



######################## Debug
if debug == 10:
    def test(sequence):
        min_dist = sys.maxsize
        label = None
        for key, values in ideal_sequences.items():
            dist = LevenshteinDistance.levenshtein(ideal_sequences[key], sequence)
            if dist < min_dist:
                min_dist = dist
                label = key
        return label

    ideal_sequences = {
        #'arrow'                     : ['O', 'A1', 'O', 'A4', 'O', 'A0', 'O', 'A5', 'O'],
        #'caret'                     : ['O', 'A1', 'O', 'A7', 'O'],
        #'delete_mark'               : ['O', 'A7', 'O', 'A4', 'O', 'A1', 'O'],
        'left_sq_bracket'           : ['O', 'A4', 'O', 'A6', 'O', 'A0', 'O'],
        #'rectangle'                 : ['O', 'A6', 'O', 'A0', 'O', 'A2', 'O', 'A4', 'O'],
        #'right_sq_bracket'          : ['O', 'A0', 'O', 'A6', 'O', 'A4', 'O'],
        #'star'                      : ['O', 'A1', 'O', 'A7', 'O', 'A3', 'O', 'A0', 'O', 'A5', 'O'],
        #'triangle'                  : ['O', 'A5', 'O', 'A0', 'O', 'A3', 'O'],
        'v'                         : ['O', 'A7', 'O', 'A1', 'O'],
        #'x'                         : ['O', 'A7', 'O', 'A2', 'O', 'A5', 'O']
    }
    directories = [key for key in ideal_sequences]

    # matrix
    result = ConfusionMatrix(gesture_labels=directories)
    for directory in directories:
        # original kalman + resampled
        dataset = transforms_set(directory, parsing=True)
        # apply transforms
        for sequence in dataset.applyTransforms():
            print(sequence)
            column_label = test(sequence[0])
            result.update(row_label=directory, column_label=column_label)
    result.plot()


if debug == 11:
    seq = list('CGACTACTGACTACTCGCCGACGCGACTGCCGTCTATACTGCGCATACGGC')

    d1 = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})
    d2 = DiscreteDistribution({'A': 0.10, 'C': 0.40, 'G': 0.40, 'T': 0.10})

    s1 = State(d1, name='background')
    s2 = State(d2, name='CG island')

    gmm = GeneralMixtureModel([d1, d2])
    hmm = HiddenMarkovModel()

    hmm.add_states(s1, s2)
    hmm.add_transition(hmm.start, s1, 0.5)
    hmm.add_transition(hmm.start, s2, 0.5)
    hmm.add_transition(s1, s1, 0.89)
    hmm.add_transition(s1, s2, 0.1)
    hmm.add_transition(s1, hmm.end, 0.01)
    hmm.add_transition(s2, s1, 0.1)
    hmm.add_transition(s2, s2, 0.9)
    hmm.bake()

    gmm_predictions = gmm.predict(np.array(seq))
    hmm_predictions = hmm.predict(seq)

    print ("sequence: {}".format(''.join(seq)))
    print ("gmm pred: {}".format(''.join(map(str, gmm_predictions))))
    print ("hmm pred: {}".format(''.join(map(str, hmm_predictions))))

    print('\nhmm state 0: {}'.format(hmm.states[0].name))
    print('hmm state 1: {}'.format(hmm.states[1].name))

    print (hmm.predict_proba(seq))

# sequence alignment
if debug == 12:
    ### V model ###
    model_v = HiddenMarkovModel("v")

    # Define the distribution for insertions
    i_d = DiscreteDistribution(
        {'A0': 0.1111, 'A1': 0.1111, 'A2': 0.1111, 'A3': 0.1111, 'A4': 0.1111,
         'A5': 0.1111, 'A6': 0.1111, 'A7': 0.1111, 'O': 0.1112}
    )

    # Create the insert states
    i0 = State(i_d, name="I0")
    i1 = State(i_d, name="I1")
    i2 = State(i_d, name="I2")
    i3 = State(i_d, name="I3")
    i4 = State(i_d, name="I4")
    i5 = State(i_d, name="I5")

    # Create the match states
    m1 = State(DiscreteDistribution({"A0": 0.01, "A1": 0.01, "A2": 0.01, "A3": 0.01, "A4": 0.01, "A5": 0.01, "A6": 0.01, "A7": 0.01, "O": 0.92}), name="M1")
    m2 = State(DiscreteDistribution({"A0": 0.20, "A1": 0.01, "A2": 0.01, "A3": 0.01, "A4": 0.01, "A5": 0.01, "A6": 0.20, "A7": 0.54, "O": 0.01}), name="M2")
    m3 = State(DiscreteDistribution({"A0": 0.01, "A1": 0.01, "A2": 0.01, "A3": 0.01, "A4": 0.01, "A5": 0.01, "A6": 0.01, "A7": 0.01, "O": 0.92}), name="M3")
    m4 = State(DiscreteDistribution({"A0": 0.01, "A1": 0.92, "A2": 0.01, "A3": 0.01, "A4": 0.01, "A5": 0.01, "A6": 0.01, "A7": 0.01, "O": 0.01}), name="M4")
    m5 = State(DiscreteDistribution({"A0": 0.01, "A1": 0.01, "A2": 0.01, "A3": 0.01, "A4": 0.01, "A5": 0.01, "A6": 0.01, "A7": 0.01, "O": 0.92}), name="M5")

    # Create delete states
    d1 = State(None, name="D1")
    d2 = State(None, name="D2")
    d3 = State(None, name="D3")
    d4 = State(None, name="D4")
    d5 = State(None, name="D5")

    # Add all the states to the model
    model_v.add_states([i0, i1, i2, i3, i4, i5, m1, m2, m3, m4, m5, d1, d2, d3, d4, d5])

    # Create transitions from match states
    model_v.add_transition(model_v.start, m1, 0.9)
    model_v.add_transition(model_v.start, i0, 0.1)

    model_v.add_transition(m1, m2, 0.9)
    model_v.add_transition(m1, i1, 0.05)
    model_v.add_transition(m1, d2, 0.05)

    model_v.add_transition(m2, m3, 0.9)
    model_v.add_transition(m2, i2, 0.05)
    model_v.add_transition(m2, d3, 0.05)

    model_v.add_transition(m3, m4, 0.9)
    model_v.add_transition(m3, i3, 0.05)
    model_v.add_transition(m3, d4, 0.05)

    model_v.add_transition(m4, m5, 0.9)
    model_v.add_transition(m4, i4, 0.05)
    model_v.add_transition(m4, d5, 0.05)

    model_v.add_transition(m5, model_v.end, 0.9)
    model_v.add_transition(m5, i5, 0.1)

    # Create transitions from insert states
    model_v.add_transition(i0, i0, 0.70)
    model_v.add_transition(i0, d1, 0.15)
    model_v.add_transition(i0, m1, 0.15)

    model_v.add_transition(i1, i1, 0.70)
    model_v.add_transition(i1, d2, 0.15)
    model_v.add_transition(i1, m2, 0.15)

    model_v.add_transition(i2, i2, 0.70)
    model_v.add_transition(i2, d3, 0.15)
    model_v.add_transition(i2, m3, 0.15)

    model_v.add_transition(i3, i3, 0.70)
    model_v.add_transition(i3, d4, 0.15)
    model_v.add_transition(i3, m4, 0.15)

    model_v.add_transition(i4, i4, 0.70)
    model_v.add_transition(i4, d5, 0.15)
    model_v.add_transition(i4, m5, 0.15)

    model_v.add_transition(i5, i5, 0.85)
    model_v.add_transition(i5, model_v.end, 0.15)

    # Create transitions from delete states
    model_v.add_transition(d1, d2, 0.15)
    model_v.add_transition(d1, i1, 0.15)
    model_v.add_transition(d1, m2, 0.70)

    model_v.add_transition(d2, d3, 0.15)
    model_v.add_transition(d2, i2, 0.15)
    model_v.add_transition(d2, m3, 0.70)

    model_v.add_transition(d3, d4, 0.15)
    model_v.add_transition(d3, i3, 0.15)
    model_v.add_transition(d3, m4, 0.70)

    model_v.add_transition(d4, d5, 0.15)
    model_v.add_transition(d4, i4, 0.15)
    model_v.add_transition(d4, m5, 0.70)

    model_v.add_transition(d5, i5, 0.30)
    model_v.add_transition(d5, model_v.end, 0.70)

    # Call bake to finalize the structure of the model.
    model_v.bake()







    ##### Left sq bracket ####
    model_left_sq_bracket = HiddenMarkovModel("left_sq_bracket")

    # Define the distribution for insertions
    i_d = DiscreteDistribution(
        {'A0': 0.1111, 'A1': 0.1111, 'A2': 0.1111, 'A3': 0.1111, 'A4': 0.1111,
         'A5': 0.1111, 'A6': 0.1111, 'A7': 0.1111, 'O': 0.1112}
    )
    # Create the insert states
    i0 = State(i_d, name="I0")
    i1 = State(i_d, name="I1")
    i2 = State(i_d, name="I2")
    i3 = State(i_d, name="I3")
    i4 = State(i_d, name="I4")
    i5 = State(i_d, name="I5")
    i6 = State(i_d, name="I6")
    i7 = State(i_d, name="I7")

    # Create the match states
    m1 = State(DiscreteDistribution({"A0": 0.01, "A1": 0.01, "A2": 0.01, "A3": 0.01, "A4": 0.01, "A5": 0.01, "A6": 0.01, "A7": 0.01, "O": 0.92}), name="M1")
    m2 = State(DiscreteDistribution({"A0": 0.01, "A1": 0.01, "A2": 0.01, "A3": 0.01, "A4": 0.92, "A5": 0.01, "A6": 0.01, "A7": 0.01, "O": 0.01}), name="M2")
    m3 = State(DiscreteDistribution({"A0": 0.01, "A1": 0.01, "A2": 0.01, "A3": 0.01, "A4": 0.01, "A5": 0.01, "A6": 0.01, "A7": 0.01, "O": 0.92}), name="M3")
    m4 = State(DiscreteDistribution({"A0": 0.01, "A1": 0.01, "A2": 0.01, "A3": 0.01, "A4": 0.01, "A5": 0.01, "A6": 0.92, "A7": 0.01, "O": 0.01}), name="M4")
    m5 = State(DiscreteDistribution({"A0": 0.01, "A1": 0.01, "A2": 0.01, "A3": 0.01, "A4": 0.01, "A5": 0.01, "A6": 0.01, "A7": 0.01, "O": 0.92}), name="M5")
    m6 = State(DiscreteDistribution({"A0": 0.92, "A1": 0.01, "A2": 0.01, "A3": 0.01, "A4": 0.01, "A5": 0.01, "A6": 0.01, "A7": 0.01, "O": 0.01}), name="M6")
    m7 = State(DiscreteDistribution({"A0": 0.01, "A1": 0.01, "A2": 0.01, "A3": 0.01, "A4": 0.01, "A5": 0.01, "A6": 0.01, "A7": 0.01, "O": 0.92}), name="M7")

    # Create delete states
    d1 = State(None, name="D1")
    d2 = State(None, name="D2")
    d3 = State(None, name="D3")
    d4 = State(None, name="D4")
    d5 = State(None, name="D5")
    d6 = State(None, name="D6")
    d7 = State(None, name="D7")

    # Add all the states to the model
    model_left_sq_bracket.add_states([i0, i1, i2, i3, i4, i5, i6, i7, m1, m2, m3, m4, m5, m6, m7, d1, d2, d3, d4, d5, d6, d7])

    # Create transitions from match states
    model_left_sq_bracket.add_transition(model_left_sq_bracket.start, m1, 0.9)
    model_left_sq_bracket.add_transition(model_left_sq_bracket.start, i0, 0.1)

    model_left_sq_bracket.add_transition(m1, m2, 0.9)
    model_left_sq_bracket.add_transition(m1, i1, 0.05)
    model_left_sq_bracket.add_transition(m1, d2, 0.05)

    model_left_sq_bracket.add_transition(m2, m3, 0.9)
    model_left_sq_bracket.add_transition(m2, i2, 0.05)
    model_left_sq_bracket.add_transition(m2, d3, 0.05)

    model_left_sq_bracket.add_transition(m3, m4, 0.9)
    model_left_sq_bracket.add_transition(m3, i3, 0.05)
    model_left_sq_bracket.add_transition(m3, d4, 0.05)

    model_left_sq_bracket.add_transition(m4, m5, 0.9)
    model_left_sq_bracket.add_transition(m4, i4, 0.05)
    model_left_sq_bracket.add_transition(m4, d5, 0.05)

    model_left_sq_bracket.add_transition(m5, m6, 0.9)
    model_left_sq_bracket.add_transition(m5, i5, 0.05)
    model_left_sq_bracket.add_transition(m5, d6, 0.05)

    model_left_sq_bracket.add_transition(m6, m7, 0.9)
    model_left_sq_bracket.add_transition(m6, i6, 0.05)
    model_left_sq_bracket.add_transition(m6, d7, 0.05)

    model_left_sq_bracket.add_transition(m7, model_left_sq_bracket.end, 0.9)
    model_left_sq_bracket.add_transition(m7, i7, 0.1)

    # Create transitions from insert states
    model_left_sq_bracket.add_transition(i0, i0, 0.70)
    model_left_sq_bracket.add_transition(i0, d1, 0.15)
    model_left_sq_bracket.add_transition(i0, m1, 0.15)

    model_left_sq_bracket.add_transition(i1, i1, 0.70)
    model_left_sq_bracket.add_transition(i1, d2, 0.15)
    model_left_sq_bracket.add_transition(i1, m2, 0.15)

    model_left_sq_bracket.add_transition(i2, i2, 0.70)
    model_left_sq_bracket.add_transition(i2, d3, 0.15)
    model_left_sq_bracket.add_transition(i2, m3, 0.15)

    model_left_sq_bracket.add_transition(i3, i3, 0.70)
    model_left_sq_bracket.add_transition(i3, d4, 0.15)
    model_left_sq_bracket.add_transition(i3, m4, 0.15)

    model_left_sq_bracket.add_transition(i4, i4, 0.70)
    model_left_sq_bracket.add_transition(i4, d5, 0.15)
    model_left_sq_bracket.add_transition(i4, m5, 0.15)

    model_left_sq_bracket.add_transition(i5, i5, 0.70)
    model_left_sq_bracket.add_transition(i5, d6, 0.15)
    model_left_sq_bracket.add_transition(i5, m6, 0.15)

    model_left_sq_bracket.add_transition(i6, i6, 0.70)
    model_left_sq_bracket.add_transition(i6, d7, 0.15)
    model_left_sq_bracket.add_transition(i6, m7, 0.15)

    model_left_sq_bracket.add_transition(i7, i7, 0.85)
    model_left_sq_bracket.add_transition(i7, model_left_sq_bracket.end, 0.15)

    # Create transitions from delete states
    model_left_sq_bracket.add_transition(d1, d2, 0.15)
    model_left_sq_bracket.add_transition(d1, i1, 0.15)
    model_left_sq_bracket.add_transition(d1, m2, 0.70)

    model_left_sq_bracket.add_transition(d2, d3, 0.15)
    model_left_sq_bracket.add_transition(d2, i2, 0.15)
    model_left_sq_bracket.add_transition(d2, m3, 0.70)

    model_left_sq_bracket.add_transition(d3, d4, 0.15)
    model_left_sq_bracket.add_transition(d3, i3, 0.15)
    model_left_sq_bracket.add_transition(d3, m4, 0.70)

    model_left_sq_bracket.add_transition(d4, d5, 0.15)
    model_left_sq_bracket.add_transition(d4, i4, 0.15)
    model_left_sq_bracket.add_transition(d4, m5, 0.70)

    model_left_sq_bracket.add_transition(d5, d6, 0.15)
    model_left_sq_bracket.add_transition(d5, i5, 0.15)
    model_left_sq_bracket.add_transition(d5, m6, 0.70)

    model_left_sq_bracket.add_transition(d6, d7, 0.15)
    model_left_sq_bracket.add_transition(d6, i6, 0.15)
    model_left_sq_bracket.add_transition(d6, m7, 0.70)

    model_left_sq_bracket.add_transition(d7, i7, 0.30)
    model_left_sq_bracket.add_transition(d7, model_left_sq_bracket.end, 0.70)

    # Call bake to finalize the structure of the model.
    model_left_sq_bracket.bake()


    # add models to list
    models = [model_left_sq_bracket, model_v]


    ### Test ###
    # test 1
    # for sequence in map(list, (['O', 'A1', 'O', 'A4', 'O', 'A0', 'O', 'A5', 'O'], ['O', 'A1', 'O', 'A7', 'O'],
    #                            ['O', 'A7', 'O', 'A4', 'O', 'A1', 'O'], ['O', 'A4', 'O', 'A6', 'O', 'A0', 'O'])):
    #     for m in models:
    #         logp, path = m.viterbi(sequence)
    #         print ("Model {} - Sequence: '{}'  -- Log Probability: {} -- Path: {}".format(
    #             m.name, ''.join(sequence), logp, " ".join(state.name for idx, state in path[1:-1])))

    # test 2
    directories = [m.name for m in models]
    result = ConfusionMatrix(directories)
    for directory in directories:
        # original kalman + resampled + parsing
        dataset = transforms_set(directory, parsing=True)

        # apply transforms
        for sequence in dataset.applyTransforms():
            column_label = Test.compare(sequence=sequence[0], gesture_hmms=models)
            result.update(row_label=directory, column_label=column_label)
            if column_label != directory:
                print(sequence)
                print(column_label + '\n')
    result.plot()


