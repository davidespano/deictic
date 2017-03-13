from dataset import *
from gesture import *
from test import *

# Test
def deictic_test(gestureDir, gesture_models, n_states=6, n_samples=20, plot=False):
    hmms = []
    names = []
    for gesture in gesture_models:
        factory = ClassifierFactory()
        factory.setLineSamplesPath(trainingDir)
        factory.setClockwiseArcSamplesPath(arcClockWiseDir)
        factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
        factory.states = n_states
        factory.spu = n_samples

        model, edges = factory.createClassifier(gesture[0])
        hmms.append(model)
        names.append(gesture[1])

    # Compare
    results = compares_deictic_models(hmms, gestureDir, names, plot=plot)
    list_models, list_names = zip(*gesture_models)
    print(list_names)
    print(results)

# Test ad-hoc hidden markov models
def adhoc_test(gestureDir, list_gesture, dimensions=2, scale=100):

    # Results
    results = numpy.zeros((len(list_gesture), len(list_gesture)))
    list_dataset = []
    for gesture in list_gesture:
        list_dataset.append(CsvDataset(gestureDir+gesture[0]+'/'))

    # Training
    len_sequence = len(list_dataset[0].read_dataset())
    for index in range(0, 1):# len_sequence):
        list_testing = []
        # Create hmm gesture, training and testing sequences
        models = []
        index_dataset = 0
        for gesture in list_gesture:
            # Gets traning and testing sequences
            te_seq , tr_seq = list_dataset[index_dataset].leave_one_out(index)
            # Create and training hmm
            models.append(create_hmm_gesture(gesture[0], tr_seq, gesture[1]))
            # Test list
            list_testing.append(te_seq)
            # Index dataset
            index_dataset = index_dataset+1

        # Testing
        results = compares_adhoc_models(models, list_testing, gestureDir, results)

    return results



# Main
baseDir = '/home/alessandro/PycharmProjects/deictic/repository/'
n_states = 6 # Numero stati
n_samples = 40
mode = 1

baseDir  = '/Users/davide/Google Drive/Dottorato/Software/python/hmmtest/repository/'
#baseDir = '/home/alessandro/PycharmProjects/deictic/repository/'
trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'
testDir = baseDir + "deictic/mdollar-dataset/resampled/"

################################################################ DEICTIC HMM ################################################################
# 1Dollar Test
if mode == 0:
    gestureDir = baseDir + 'deictic/1dollar-dataset/resampled/'
    gesture_models = [
        (Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4), 'triangle'), # triangle
        (Point(0,0) + Line(3,-3) + Line(0,3) + Line(-3,-3), 'x'), # X
        (Point(0,0) + Line(0,-3) + Line(4,0) + Line(0, 3) + Line(-4,0), 'rectangle'), # rectangle
        (Point(0,0) + Arc(-3,-3, cw=False) + Arc(3,-3, cw=False) + Arc(3,3, cw=False) + Arc(-3,3, cw=False), 'circle'), # circle
        (Point(0,0) + Line(2, -2) + Line(4,6), 'check'), # check
        (Point(0,0) + Line(2,3) + Line(2,-3), 'caret'), # caret
        (Point(0,0) + Arc(2,2) + Arc(2,-2) + Arc(-2,-2) + Line(0,-3), 'question_mark'), # question mark
        (Point(0,0) + Line(6,4) + Line(-4,0) + Line(5,1) + Line(-1, -4), 'arrow'),
        (Point(0,0) + Line(-2,0) + Line(0,-4) + Line(2,0), 'left_sq_bracket'), # left square bracket
        (Point(0,0) + Line(2,0) + Line(0, -4)  + Line(-2, 0), 'right_sq_bracket'), # right square bracket
        (Point(0,0) + Line(2,-3) + Line(2,3), 'v'), # V
        (Point(0,0) + Line(2, -3) + Line(-2,0) + Line(2,3), 'delete_mark'), # delete
        (Point(0,0) + Arc(-5,-5, cw=False) + Line(0,-6) + Arc(-3,-3)  + Arc(3,-3) + Line(0,-6) + Arc(5,-5,cw=False), "left_curly_brace"), # left curly brace
        (Point(0,0) + Arc(5,-5) + Line(0,-6) + Arc(3,-3, cw=False) + Arc(-3,-3, cw=False) + Line(0,-6) + Arc(-5,-5), "right_curly_brace"),  # right curly brace
        (Point(0,0) + Line(2,5) + Line(2, -5) + Line(-5, 3) + Line(6,0) + Line(-5, -3), 'star'), # star
        (Point(0,0) + Arc(3,3, cw=False) + Arc(-1,1, cw=False) + Arc(-1,-1, cw=False) + Arc(3, -3, cw=False), "pigtail") # pigtail
    ]

    results = deictic_test(gestureDir, gesture_models, n_states, plot=False)

# MDollar Test
if mode == 1:
    gestureDir = baseDir + 'deictic/mdollar-dataset/resampled/'
    gesture_models = [
        ((Point(0, 0) + Line(6, 0) + Point(4, 2) + Line(2, -2) + Line(-2, -2)), 'arrowhead'),  # arrowhead
        ((Point(0, 4) + Line(0, -4) + Point(-1, 2) + Line(5, 0) + Point(4, 4) + Line(0, -4)), 'H'),  # H
        ((Point(0, 4) + Line(0, -4) + Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(0, -4)), 'N'),  # N
        ((Point(0, 4) + Line(4, 0) + Point(2, 4) + Line(0, -4) + Point(0, 0) + Line(4, 0)), 'I'),  # I
        ((Point(0, 0) + Line(0, -4) + Point(0, 0) + Arc(1, -1, cw=True) + Arc(-1, -1, cw=True)), 'P'),  # P
        ((Point(-2, 0) + Line(4, 0) + Point(0, 0) + Line(0, -4) ), 'T'),  # T
        ((Point(0, 0.5) + Line(2, 2) + Line(2, -2) + Line(-4, 0) + Point(0, 2) + Line(4, 0) + Line(-2, -2) + Line(-2, 2)), 'six_point_star'),  # six point star
        ((Point(0, 0) + Line(0, 4) + Point(0, 4) + Arc(2, -2, cw=True) + Point(2, 2) + Arc(-2, -2, cw=True)), 'D'),  # D
        ((Point(4,3) + Line(-4,-3) + Point(0,3) + Line (4,-3) + Point(2,4) + Line(0, -4)), 'asterisk'),  # asterisk
        ((Point(0, 20) + Line(0, -19)+ Point(0, 1) + Line(0, -1) ), 'exclamation_point'),  # exclamatoin point
        ((Point(0,0) + Arc(-3,-3, cw=False) + Arc(3,-3, cw=False) + Arc(3,3, cw=False) + Arc(-3,3, cw=False) +
          Point(4,1) + Line(-8, -8)), 'null'), # Null
        ((Point(-2,4)+Arc(2,-2, cw=False) + Point(0,2)+Arc(2,2, cw=False) + Point(0,4)+Line(0,-4)), 'pitchfork'),# pitchfork
        ((Point(0,0)+Line(0,-4) + Point(0,-4)+Arc(-1,-1, cw=False) + Point(-1,-5)+Arc(1,1, cw=False)), 'half_note'),# half note
        ((Point(4,3) + Line(-4,-3) + Point(0,3) + Line (4,-3)), 'X')# X
    ]
    results = deictic_test(gestureDir, gesture_models, n_states, plot=False)

################################################################ DEICTIC COMPOSED HMM ################################################################
if mode == 2 or mode == 3:
    if mode == 2:
        gestureDir = baseDir + 'deictic/1dollar-dataset/synthetic/prova/'
    else:
        gestureDir = baseDir + 'deictic/mdollar-dataset/synthetic/'

    folders = [name for name in os.listdir(gestureDir) if os.path.isdir(os.path.join(gestureDir, name))]
    hmms = []
    parse = Parse(n_states, n_samples)
    for folder in folders:
        model = parse.parse_expression_2(folder)
        hmms.append(model)

    compares_deictic_models(hmms, gestureDir, folders, plot=False)



################################################################ ADHOC HMM ################################################################
## Adhoc hmm - 1Dollar
if mode == 4:
    list_gesture = [("rectangle", n_states*4), ("triangle", n_states*3), ("caret", n_states*2), ("v", n_states*2), ("x", n_states*3),
                    ("left_sq_bracket", n_states*3), ("right_sq_bracket", n_states*3), ("delete_mark", n_states*4), ("star", n_states*4),
                    ("arrow", n_states*4), ("check", n_states*2), ("circle", n_states*4), ("left_curly_brace", n_states*6),
                    ("right_curly_brace", n_states*6), ("pigtail", n_states*4), ("question_mark", n_states*4)]
    results = adhoc_test(gestureDir, list_gesture)
## Adhoc hmm - MDollar
if mode == 5:
    list_gesture = [("D", n_states*3), ("H", n_states*3), ("I", n_states*3), ("N", n_states*3), ("P", n_states*3),
                    ("T", n_states*2), ("X", n_states*2), ("arrowhead", n_states*3), ("asterisk", n_states*4),
                    ("exclamation_point", n_states*2), ("half_note", n_states*3),
                    ("null", n_states*5), ("pitchfork", n_states*3), ("six_point_star", n_states*6)]
    results = adhoc_test(gestureDir, list_gesture)









if mode == 24:
    #t = Point(0,0) + Line(4,0) + Point(2,0) + Line(0, -4)
    #t = Point(0,0) + Line(6,4) + Line(-4,0) + Line(5,1) + Line(-1, -4)
    # t.plot()
    gesture_models[10][0].plot()
    factory = ClassifierFactory()
    factory.setLineSamplesPath(trainingDir)
    factory.setClockwiseArcSamplesPath(arcClockWiseDir)
    factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
    model, edges = factory.createClassifier(gesture_models[10][0])
    plot_gesture(model)

if mode == 25:
    dataset = CsvDataset(testDir + "exclamation_point/")
    #dataset.plot()
    dataset.plot(singleMode=True)

if mode == 26:
    d = NormalDistribution(2.0, 0.5)
    samples = [d.sample() for i in range(10000)]
    plt.hist(samples, edgecolor='c', color='c', bins=50)
    plt.show()