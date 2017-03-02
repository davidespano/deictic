from dataset import *
from gesture import *
from test import *

# Test
def deictic_test(gestureDir, gesture_models, n_states=6, n_samples=20, plot=False):

    # Folders
    trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
    arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
    arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'

    hmms = []
    names = []
    for gesture in gesture_models:
        factory = ClassifierFactory()
        factory.setLineSamplesPath(trainingDir)
        factory.setClockwiseArcSamplesPath(arcClockWiseDir)
        factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
        factory.spu = n_states
        factory.spu = n_samples

        model, edges = factory.createClassifier(gesture[0])
        hmms.append(model)
        names.append(gesture[1])

    # Compare
    results = compares_deictic_models(hmms, gestureDir, names, plot=plot)
    list_models, list_names = zip(*gesture_models)
    print(list_names)
    print(results)

def deictic_test_unistroke(primitiveDir, gestureDir, n_states):
    models = []

    #models.append(create_arrow(primitiveDir, n_states)[0])
    models.append(create_caret(primitiveDir, n_states)[0])
    models.append(create_delete(primitiveDir, n_states)[0])
    models.append(create_rectangle(primitiveDir, n_states)[0])
    models.append(create_square_braket_left(primitiveDir, n_states)[0])
    models.append(create_square_braket_right(primitiveDir, n_states)[0])
    models.append(create_star(primitiveDir, n_states)[0])
    models.append(create_triangle(primitiveDir, n_states)[0])
    models.append(create_v(primitiveDir, n_states)[0])
    models.append(create_x(primitiveDir, n_states)[0])

    #models.append(primitive_model(primitiveDir, n_states, Primitive.right))
    #models.append(primitive_model(primitiveDir, n_states, Primitive.left))

    # Compare
    names = {'arrow', 'caret', 'delete_mark', 'rectangle', 'left_sq_bracket',
             'right_sq_bracket', 'star', 'triangle', 'v', 'x'}
    return compares_deictic_models(models, gestureDir, names)

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
n_states = 8 # Numero stati
n_samples = 40
mode = 1

## Deictic
if mode == 0:
    gestureDir = baseDir+'deictic/1dollar-dataset/resampled/'

    gesture_models = [
        (Point(0,0) + Line(-2,-3) + Line(4,0)+ Line(-2,3), 'triangle'), # triangle
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
        (Point(0,0) + Arc(-2,-2, cw=False) + Line(0,-3) + Arc(-1,-1)  + Arc(1,-1) + Line(0,-3) + Arc(2,-2,cw=False), "left_curly_brace"), # left curly brace
        (Point(0,0) + Arc(2,-2) + Line(0,-3) + Arc(1,-1, cw=False) + Arc(-1,-1, cw=False) + Line(0,-3) + Arc(-2,-2), "right_curly_brace"),  # right curly brace
        (Point(0,0) + Line(2,5) + Line(2, -5) + Line(-5, 3) + Line(6,0) + Line(-5, -3), 'star'), # star
        (Point(0,0) + Arc(3,3, cw=False) + Arc(-1,1, cw=False) + Arc(-1,-1, cw=False) + Arc(3, -3, cw=False), "pigtail") # pigtail
    ]
    results = deictic_test(gestureDir, gesture_models, n_states=n_states, n_samples=n_samples, plot=False)
    #results = deictic_test_unistroke(primitiveDir, gestureDir, n_states)

# Multistroke
if mode == -1:
    gestureDir = baseDir + 'deictic/mdollar-dataset/resampled/'

    gesture_models = [
        # Pitchfork
        ( (Point(0,0)+Line(2,3) + Line(2,-3)+Line(-4,2) + Line(4,0)+Line(-4, -2)), 'five_point_star'),
    ]
    #gesture_models[0][0].plot()
    results = deictic_test(gestureDir, gesture_models, n_states, plot=True)
if mode == 1:
    gestureDir = baseDir + 'deictic/mdollar-dataset/resampled/'

    gesture_models = [
        ( (Point(0,0)+Line(6,0) + Point(4,2)+Line(2,-2) + Line(-2,-2)), 'arrowhead'),# arrowhead
        ( ( Point(0,4)+Line(0,-4) + Point(-1,2)+Line(5,0) + Point(4,4)+Line(0,-4) ), 'H'),# H
        ( (Point(0,4)+Line(0,-4) + Point(0,4)+Line(4,-4) + Point(4,4)+Line(0,-4)), 'N' ),# N
        ( (Point(0,4)+Line(4,0) + Point(2,4)+Line(0,-4) + Point(0,0)+Line(4,0)), 'I'),# I
        ( ( Point(0,0)+Line(0,4) + Point(0,4)+Arc(1,-1, cw=True) + Point(1,3)+Arc(-1,-1, cw=True)), 'P'),# P
        ( (Point(0,0)+Line(4,0) + Point(2,0)+Line(0,-4)), 'T' ),# T
        ( (Point(0,0)+Line(2,4) + Line(2,-4)+Line(-4,2) + Line(4,0)+Line(-4, -2)), 'five_point_star'), # five point star
        ( (Point(0,0.5)+Line(2,2)+Line(2,-2)+Line(-4,0) + Point(0,2)+Line(4,0)+Line(-2,-2)+Line(-2,2)), 'six_point_star'),# six point star
        ( ( Point(0,0)+Line(0,4) + Point(0,4)+Arc(2,-2, cw=True) + Point(2,2)+Arc(-2,-2, cw=True)), 'D'),# D
        ( (Point(0,0)+Line(0,4) + Point(2,0)+Line(-4,4) + Point(-2,0)+Line(4,4) ), 'asterisk'), # asterisk
        ( (Point(0,0)+Line(0.1,0.1) + Point(0,1)+Line(0,3)), 'exclamation_point'), # exclamatoin point
        # Line - crash
        #( (Point(0,0)+Line(4,0) + Point(4,0)+Line(10,0)), 'line'),
        # Null - 0%
        #( ( Point(-4,-4)+Line(8,8) +
        #    Point(0,0) + Arc(-3,-3, cw=False) + Arc(3,-3, cw=False) + Arc(3,3, cw=False) + Arc(-3,3, cw=False)), 'null'),

        #( ( Point(-2,4)+Arc(2,-2, cw=False) + Point(0,2)+Arc(2,2, cw=False) + Point(0,4)+Line(0,-4)), 'pitchfork'),# pitchfork
        #( ( Point(0,0)+Line(0,-4) + Point(0,-4)+Arc(-1,-1, cw=False) + Point(-1,-5)+Arc(1,1, cw=False)), 'half_note'),# half note
        #( (Point(0,0)+Line(4,4) + Point(4,0)+Line(-4,4)), 'X')# X
    ]

    #gesture_models[0][0].plot()
    results = deictic_test(gestureDir, gesture_models, n_states, plot=False)


## Adhoc hmm
if mode == 2:
    list_gesture = [("rectangle", n_states*4), ("triangle", n_states*3), ("caret", n_states*2), ("v", n_states*2), ("x", n_states*3),
            ("left_sq_bracket", n_states*3), ("right_sq_bracket", n_states*3), ("delete", n_states*4), ("star", n_states*4),
            ("arrow", n_states*4), ("check", n_states*2), ("circle", n_states*4), ("left_curly_brace", n_states*6),
            ("right_curly_brace", n_states*6), ("pigtail", n_states*4), ("question_mark", n_states*4)]
    #results = adhoc_test(gestureDir, list_gesture)
