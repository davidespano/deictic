from dataset import *
from gesture import *
from test import *


def create_hmms(gesture_models, n_states=6, n_samples=20):
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
        for gesture in gesture_models[k]:
            model, edges = factory.createClassifier(gesture)
            hmms[k].append(model)
    return hmms;

# Main
#baseDir = '/home/alessandro/PycharmProjects/deictic/repository/'
baseDir  = '/Users/davide/PycharmProjects/deictic/repository/'
trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'
testDir = baseDir + "deictic/mdollar-dataset/resampled/"
outputDir = baseDir +"deictic/mdollar-dataset/ten-cross-validation/"

n_states = 6 # Numero stati
n_samples = 20
mode = 3

gesture_models = {
    'T': [
        Point(-2, 0) + Line(4, 0) + Point(0, 0) + Line(0, -4),
        #Point(-2, 0) + Line(4, 0) + Point(-4, 2) + Line(0, 4),
        #Point(2, 0) + Line(-4, 0) + Point(0, 0) + Line(0, -4),
        #Point(2, 0) + Line(-4, 0) + Point(-4, 2) + Line(0, 4),
        Point(0, 0) + Line(0, -4) + Point(-2, 0) + Line(4, 0),
        #Point(0, 0) + Line(0, -4) + Point(2, 0) + Line(-4, 0),
        #Point(-4, 2) + Line(0, 4) + Point(-2, 0) + Line(4, 0),
        #Point(-4, 2) + Line(0, 4) + Point(2, 0) + Line(-4, 0)
    ],

    'N': [
        (Point(0, 4) + Line(0, -4) + Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(0, -4))
    ],

    'D': [
        #Point(0, 0) + Line(0, 6) + Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0),
        Point(0, 6) + Line(0, -6) + Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0),
        #Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0) + Point(0, 0) + Line(0, 6),
        #Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0) + Point(0, 6) + Line(0, -6)
    ],

    'P': [
        #Point(0, 0) + Line(0, 8) + Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0),
        Point(0, 8) + Line(0, -8) + Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0),
        #Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0) + Point(0, 8) + Line(0, -8),
        #Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0) + Point(0, 0) + Line(0, 8)
    ],

    'X': [
        # (Point(0, 0) + Line(4, 4) + Point(4, 0) + Line(-4, 4)), NO
        (Point(0, 0) + Line(4, 4) + Point(0, 4) + Line(4, -4)),  # 33
        # (Point(4, 4) + Line(-4, -4) + Point(4, 0) + Line(-4, 4)), NO
        (Point(4, 4) + Line(-4, -4) + Point(0, 4) + Line(4, -4)),  # 174
        # (Point(4, 0) + Line(-4, 4) + Point(0, 0) + Line(4, 4)), # NO
        # (Point(4, 0) + Line(-4, 4) + Point(4, 4) + Line(-4, -4)), #NO
        (Point(0, 4) + Line(4, -4) + Point(0, 0) + Line(4, 4)), # 44
        (Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(-4, -4))  # 368
    ],

    'H': [
        (Point(0, 4) + Line(0, -4) + Point(0, 2) + Line(4, 0) + Point(4, 4) + Line(0, -4)),
        (Point(0, 4) + Line(0, -4) + Point(4, 4) + Line(0, -4) + Point(0, 2) + Line(4, 0)),
        #(Point(4, 4) + Line(0, -4) + Point(0, 4) + Line(0, -4) + Point(0, 2) + Line(4, 0))
    ],

    'I': [
        (Point(0, 4) + Line(4, 0) + Point(2, 4) + Line(0, -4) + Point(0, 0) + Line(4, 0)),
        (Point(2, 4) + Line(0, -4) + Point(0, 0) + Line(4, 0) + Point(0, 4) + Line(4, 0)),
        (Point(0, 4) + Line(4, 0) + Point(0, 0) + Line(4, 0) + Point(2, 0) + Line(0, 4)),
        (Point(2, 4) + Line(0, -4) + Point(0, 4) + Line(4, 0) + Point(0, 0) + Line(4, 0))
    ],

    'exclamation_point': [
        Point(0, 4) + Line(0, -3) + Point(0, 1) + Line(0, -1),
    ],

    'null': [
       (Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False) + Point(4, 1) + Line(-8, -8)),  # 410
       (Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False) + Point(-4, -7) + Line(8, 8)),  # 118
    ],

    'arrowhead': [
        (Point(0, 0) + Line(6, 0) + Point(4, 2) + Line(2, -2) + Line(-2, -2)),
        (Point(4, 2) + Line(2, -2) + Line(-2, -2) + Point(0, 0) + Line(6, 0)),
    ],

     'pitchfork': [
         (Point(-2, 4) + Arc(2, -2, cw=False) + Arc(2, 2, cw=False) + Point(0, 4) + Line(0, -4)),
         (Point(0, 4) + Line(0, -4) + Point(-2, 4) + Arc(2, -2, cw=False) + Arc(2, 2, cw=False))
     ],

    'six_point_star': [
        (Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4) + Point(-2, -1) + Line(4, 0) + Line(-2, -4) + Line(-2,4)),
        (Point(-2, -1) + Line(4, 0) + Line(-2, -4) + Line(-2, 4) + Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2,4)),
        (Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0) + Point(-2, 1) + Line(4, 0) + Line(-2, -4) + Line(-2,4)),
        (Point(-2, 1) + Line(4, 0) + Line(-2, -4) + Line(-2, 4) + Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0)),
        (Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0) + Point(-2, 1) + Line(2, -4) + Line(2, 4) + Line(-4, 0)),
        (Point(-2, 1) + Line(2, -4) + Line(2, 4) + Line(-4, 0) + Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0)),
        (Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4) + Point(-2, -1) + Line(2, -4) + Line(2, 4) + Line(-4, 0)),
        (Point(-2, -1) + Line(2, -4) + Line(2, 4) + Line(-4, 0) + Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4))
    ],

    'asterisk': [
        ((Point(4, 4) + Line(-4, -4)) + Point(0, 4) + Line(4, -4) + Point(2, 4) + Line(0, -4))
    ],

    'half_note': [
        (Point(0, 0) + Arc(-3, 3, cw=False) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Point(2, 16) + Line(0, -20)),
        (Point(2, 16) + Line(0, -20) + Point(0, 0) + Arc(-3, 3, cw=False) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc( 3, 3, cw=False)),
        (Point(0, 0) + Arc(-3, 3, cw=False) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Point(2, -4) + Line(0, 20)),
        (Point(2, -4) + Line(0, 20) + Point(0, 0) + Arc(-3, 3, cw=False) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False)),
    ]
}

################################################################ DEICTIC COMPOSED HMM ################################################################
if mode == 2 or mode == 3:
    if mode == 2:
        gestureDir = baseDir + 'deictic/1dollar-dataset/synthetic/'
    else:
        gestureDir = baseDir + 'deictic/mdollar-dataset/synthetic/'

    # Gets gestures
    folders = [name for name in os.listdir(gestureDir) if os.path.isdir(os.path.join(gestureDir, name))]
    #list_hmms = []
    #parse = Parse(n_states, n_samples)

    #hmms = create_hmms(gesture_models, n_states, n_samples)
    list_hmms = dict()

    # Choice splitter
    splitter = ['choice', '-multistroke-']
    # Iterative splitter
    splitter = ['iterative', '-multistroke-']
    # Parallel splitter
    splitter = ['parallel', '-multistroke-']
    # Sequence splitter
    #splitter = ['sequence', '-multistroke-']

    factory = ClassifierFactory()
    factory.setLineSamplesPath(trainingDir)
    factory.setClockwiseArcSamplesPath(arcClockWiseDir)
    factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
    factory.states = n_states
    factory.spu = n_samples

    folders = [
        "parallel-multistroke-asterisk-multistroke-N",
        "parallel-multistroke-T-multistroke-half_note",
        "parallel-multistroke-null-multistroke-H",
        "parallel-multistroke-I-multistroke-D",
        "parallel-multistroke-six_point_star-multistroke-P",

    ]

    for folder in folders:


        # Splitting
        gestures = folder.split(splitter[0])[1].split(splitter[1])

        if "iterative" in splitter:
            list_hmms[folder] = []
            for model in hmms.get(gestures[1]):
                m, s = HiddenMarkovModelTopology.iterative(model, [])
                list_hmms[folder].append(m)
        else:
            list_hmms[folder] = []
            if "sequence" in splitter:
                for first in gesture_models[gestures[1]]:
                    for second in gesture_models[gestures[2]]:
                        seq = first + second
                        model, edges = factory.createClassifier(seq)
                        list_hmms[folder].append(model)

            if "parallel" in splitter:
                 for first in gesture_models[gestures[1]]:
                    for second in gesture_models[gestures[2]]:
                        seq = first * second
                        model, edges = factory.createClassifier(seq)
                        list_hmms[folder].append(model)
            # for model_1 in hmms.get(gestures[1]):
            #     for model_2 in hmms.get(gestures[2]):
            #         if "choice" in splitter:
            #             m,s = HiddenMarkovModelTopology.choice([model_1, model_2], [])
            #         elif "parallel" in splitter:
            #             m,s = HiddenMarkovModelTopology.parallel(model_1, model_2, [])
            #         elif "sequence" in splitter:
            #             m,s = HiddenMarkovModelTopology.sequence([model_1, model_2], [])
                        #m = gestures[1] + gestures[2];






    print(compares_deictic_models(list_hmms, gestureDir))