from dataset import *
from gesture import *
from test import *

def multistroke_test(gestureDir, gesture_models, n_states=6, n_samples=20, plot=False):
    hmms = dict()

    factory = ClassifierFactory()
    factory.setLineSamplesPath(trainingDir)
    factory.setClockwiseArcSamplesPath(arcClockWiseDir)
    factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
    factory.states = n_states
    factory.spu = n_samples

    for k in gesture_models.keys():
        print('Group {0}'.format(k))
        hmms[k] = []
        for gesture in gesture_models[k] :
            model, edges = factory.createClassifier(gesture)
            hmms[k].append(model)

    results = compares_deictic_models2(hmms, gestureDir, plot)

    print(gesture_models.keys())
    print(results)


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
n_samples = 20
mode = 1

baseDir  = '/Users/davide/PycharmProjects/deictic/repository/'
#baseDir = '/home/alessandro/PycharmProjects/deictic/repository/'
trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'
testDir = baseDir + "deictic/mdollar-dataset/resampled/"


# MDollar Test
if mode == 1:



    gestureDir = baseDir + 'deictic/mdollar-dataset/resampled/'
    gesture_models = {
        'T': [
            Point(-2, 0) + Line(4, 0) + Point(0, 0) + Line(0, -4),
            Point(-2, 0) + Line(4, 0) + Point(-4, 2) + Line (0, 4),
            Point(2, 0) + Line(-4, 0) + Point(0, 0) + Line(0, -4),
            Point(2, 0) + Line(-4, 0) + Point(-4, 2) + Line (0, 4),
            Point(0, 0) + Line(0, -4) + Point(-2, 0) + Line(4,0),
            Point(0, 0) + Line(0, -4) + Point( 2, 0) + Line(-4,0),
            Point(-4, 2) + Line(0, 4) + Point(-2, 0) + Line(4,0),
            Point(-4, 2) + Line(0, 4) + Point( 2, 0) + Line(-4,0)
        ],

        'N': [
            (Point(0, 4) + Line(0, -4) + Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(0, -4))
        ],

        'D': [
            #Point(0, 0) + Line(0, 6) + Point(0, 0) + Line(2, 0) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False) + Line(-2, 0),
            Point(0, 0) + Line(0, 6) + Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2,0),
            #Point(0, 6) + Line(0, -6) + Point(0, 0) + Line(2, 0) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False) + Line(-2, 0),
            Point(0, 6) + Line(0, -6) + Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2,0),
            #Point(0, 0) + Line(2, 0) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False) + Line(-2, 0) + Point(0, 0) + Line(0, 6),
            #Point(0, 0) + Line(2, 0) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False) + Line(-2, 0) + Point(0, 6) + Line(0, -6),
            Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0) + Point(0, 0) + Line(0, 6),
            Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0) + Point(0, 6) + Line(0, -6)
        ],

        'P': [
            #Point(0, 0) + Line(0, 8) + Point(0, 4) + Line(2, 0) + Arc(2, 2, cw=False) + Arc(-2, 2, cw=False) + Line(-2, 0),
            Point(0, 0) + Line(0, 8) + Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0),
            #Point(0, 8) + Line(0, -8) + Point(0, 4) + Line(2, 0) + Arc(2, 2, cw=False) + Arc(-2, 2, cw=False) + Line(-2, 0),
            Point(0, 8) + Line(0, -8) + Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0),
            #Point(0, 4) + Line(2, 0) + Arc(2, 2, cw=False) + Arc(-2, 2, cw=False) + Line(-2, 0) + Point(0, 8) + Line(0, -8),
            #Point(0, 4) + Line(2, 0) + Arc(2, 2, cw=False) + Arc(-2, 2, cw=False) + Line(-2, 0) + Point(0, 0) + Line(0, 8),
            Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0) + Point(0, 8) + Line(0, -8),
            Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0) + Point(0, 0) + Line(0, 8)
        ],

        'X': [
            #(Point(0, 0) + Line(4, 4) + Point(4, 0) + Line(-4, 4)), NO
            #(Point(0, 0) + Line(4, 4) + Point(0, 4) + Line(4, -4)), 33
            #(Point(4, 4) + Line(-4, -4) + Point(4, 0) + Line(-4, 4)), NO
            (Point(4, 4) + Line(-4, -4) + Point(0, 4) + Line(4, -4)), #174
            #(Point(4, 0) + Line(-4, 4) + Point(0, 0) + Line(4, 4)), # NO
            #(Point(4, 0) + Line(-4, 4) + Point(4, 4) + Line(-4, -4)), #NO
            #(Point(0, 4) + Line(4, -4) + Point(0, 0) + Line(4, 4)), # 44
            (Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(-4, -4)) # 368
        ],

        'H': [
            (Point(0, 4) + Line(0, -4) + Point(0, 2) + Line(4, 0) + Point(4, 4) + Line(0, -4)),
            (Point(0, 4) + Line(0, -4) + Point(4, 4) + Line(0, -4) + Point(0, 2) + Line(4, 0)),
            (Point(4, 4) + Line(0, -4) + Point(0, 4) + Line(0, -4) + Point(0, 2) + Line(4, 0))
        ],

        'I' : [
            (Point(0, 4) + Line(4, 0) + Point(2, 4) + Line(0, -4) + Point(0, 0) + Line(4, 0)),
            (Point(2, 4) + Line(0, -4) + Point(0, 0) + Line(4, 0) + Point(0, 4) + Line(4, 0)),
            (Point(0, 4) + Line(4, 0) + Point(0, 0) + Line(4, 0) + Point(2, 0)  + Line(0, 4)),
            (Point(2, 4) + Line(0, -4) + Point(0, 4) + Line(4, 0) + Point(0, 0) + Line(4, 0))
        ],

        'exclamation_point' : [
             Point(0, 4) + Line(0, -3) + Point(0, 1) + Line(0, -1),

         ],

        'null' : [
            (Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False) + Point(4, 1) + Line(-8, -8)), #410
            #(Point(4, 1) + Line(-8, -8) + Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False)), #NO
            (Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False) + Point(-4, -7) + Line(8, 8)), #118
            #(Point(-4, -7) + Line(8, 8) + Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False)), #4
        ],

        'arrowhead' : [
            (Point(0, 0) + Line(6, 0) + Point(4, 2) + Line(2, -2) + Line(-2, -2)),
            (Point(4, 2) + Line(2, -2) + Line(-2, -2) + Point(0, 0) + Line(6, 0) ),
        ],

        'pitchfork' : [
            (Point(-2, 4) + Arc(2, -2, cw=False) + Arc(2, 2, cw=False) + Point(0, 4) + Line(0, -4)),
            (Point(0, 4) + Line(0, -4) + Point(-2, 4) + Arc(2, -2, cw=False) + Arc(2, 2, cw=False))
        ],

        'six_point_star':[
            (Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4) + Point(-2, -1) + Line(4,0) + Line(-2,-4) + Line(-2, 4)),
            (Point(-2, -1) + Line(4,0) + Line(-2,-4) + Line(-2, 4)+Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4)),
            (Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0) + Point(-2, 1) + Line(4,0) + Line(-2,-4) + Line(-2, 4)),
            (Point(-2, 1) + Line(4,0) + Line(-2,-4) + Line(-2, 4) + Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0) ),
            (Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0) + Point(-2, 1) + Line(2,-4) + Line(2, 4) + Line(-4,0)),
            (Point(-2, 1) + Line(2,-4) + Line(2, 4) + Line(-4,0) + Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0) ),
            (Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4)+ Point(-2, -1) + Line(2,-4) + Line(2, 4) + Line(-4,0)),
            (Point(-2, -1) + Line(2, -4) + Line(2, 4) + Line(-4,0)+ Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4) )
        ],

        'asterisk' : [
            ((Point(4, 4) + Line(-4, -4)) + Point(0, 4) + Line(4, -4) + Point(2, 4) + Line(0, -4))
        ],

        'half_note' : [
            (Point(0, 0)  + Arc(-3, 3, cw=False) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False)+ Point(2, 16) + Line(0, -20)),
            (Point(2, 16) + Line(0, -20) + Point(0, 0)  + Arc(-3, 3, cw=False) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False)),
            (Point(0, 0)  + Arc(-3, 3, cw=False) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False)+ Point(2, -4) + Line(0, 20)),
            (Point(2, -4) + Line(0, 20) + Point(0, 0)  + Arc(-3, 3, cw=False) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False)),
        ]





    }
    results = multistroke_test(gestureDir, gesture_models, n_states, plot=False)

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
    T_R = Point(-2, 0) + Line(4, 0)
    T_L = Point(2, 0) + Line(-4, 0)
    T_D = Point(0, 0) + Line(0, -4)
    T_U = Point(-4, 2) + Line(0, 4)

    D_U = Point(0, 0) + Line(0, 6)
    D_D = Point(0, 6) + Line(0, -6)
    D_CCW = Point(0, 0) + Line(2, 0) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False) + Line(-2, 0)
    D_CC = Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0)

    P_U = Point(0, 0) + Line(0, 8)
    P_D = Point(0, 8) + Line(0, -8)
    P_CCW = Point(0, 4) + Line(2, 0) + Arc(2, 2, cw=False) + Arc(-2, 2, cw=False) + Line(-2, 0)
    P_CC = Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0)

    Ex_LD = Point(0, 20) + Line(0, -18)
    Ex_LU = Point(0, 2) + Line(0, 18)
    Ex_SD = Point(0, 1) + Line(0, -1)
    Ex_SU = Point(0, 0) + Line(0, 1)

    x1u = Point(0,0) + Line(4,4)
    x1d = Point(4,4) + Line(-4,-4)
    x2u = Point(4,0) + Line(-4, 4)
    x2d = Point(0,4) + Line(4, -4)

    l = [
        ((Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(-4, -4)) + Point(2,4) + Line(0,-4))
    ]

    for t in l:
        t.plot()
        # processor = ModelPreprocessor(t)
        # transform1 = CenteringTransform()
        # transform2 = NormaliseLengthTransform(axisMode=True)
        # transform3 = ScaleDatasetTransform(scale=100)
        # processor.transforms.addTranform(transform1)
        # processor.transforms.addTranform(transform2)
        # processor.transforms.addTranform(transform3)
        # print(t)
        # processor.preprocess()
        # t.plot()
    #gesture_models[10][0].plot()
    #factory = ClassifierFactory()
    #factory.setLineSamplesPath(trainingDir)
    #factory.setClockwiseArcSamplesPath(arcClockWiseDir)
    #factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
    #model, edges = factory.createClassifier(gesture_models[10][0])
    #plot_gesture(model)

if mode == 25:
    dataset = CsvDataset(testDir + "exclamation_point/")
    dataset.plot(singleMode=True)
    #dataset.plot(sampleName='94-finger-slow-I-01.csv')

if mode == 26:
    d = NormalDistribution(2.0, 0.5)
    samples = [d.sample() for i in range(10000)]
    plt.hist(samples, edgecolor='c', color='c', bins=50)
    plt.show()