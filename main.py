from dataset import *
from gesture import *
from test import *

# Test
def deictic_test(primitiveDir, gestureDir, gesture_models, n_states):

    hmms = []
    names = []
    for gesture in gesture_models:
        factory = ClassifierFactory()
        factory.setLineSamplesPath(primitiveDir)
        factory.spu = n_states

        model, edges = factory.createClassifier(gesture[0])
        hmms.append(model)
        names.append(gesture[1])

    # Compare
    return compares_deictic_models(hmms, gestureDir, names)

def deictic_test_unistroke(primitiveDir, gestureDir, n_states):
    models = []

    #models.append(create_arrow(primitiveDir, n_states)[0])
    #models.append(create_caret(primitiveDir, n_states)[0])
    #models.append(create_delete(primitiveDir, n_states)[0])
    #models.append(create_rectangle(primitiveDir, n_states)[0])
    #models.append(create_square_braket_left(primitiveDir, n_states)[0])
    #models.append(create_square_braket_right(primitiveDir, n_states)[0])
    #models.append(create_star(primitiveDir, n_states)[0])
    models.append(create_triangle(primitiveDir, n_states)[0])
    #models.append(create_v(primitiveDir, n_states)[0])
    #models.append(create_x(primitiveDir, n_states)[0])

    #models.append(primitive_model(primitiveDir, n_states, Primitive.right))
    #models.append(primitive_model(primitiveDir, n_states, Primitive.left))

    # Compare
    return compares_deictic_models(models, gestureDir)

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
gestureDir = '/home/alessandro/PycharmProjects/deictic/repository/deictic/mdollar-dataset/resampled/'
primitiveDir = '/home/alessandro/PycharmProjects/deictic/repository/deictic/unica-dataset/raw/right/'
n_states = 5 # Numero stati

mode = 1

## Deictic
# Unistroke
if mode == 0:
    results = deictic_test(primitiveDir, gestureDir, n_states)

# Multistroke
if mode == 1:
    gesture_models = [
        ((Point(0,0) + Line(4,0) + Point(2,0)+ Line(0,-4)), 'T')# T
        # N
        # X
        # I
        # Asterisk
        # Five point star
        # Six point star
    ]

    gesture_models[0][0].plot()
    results = deictic_test(primitiveDir, gestureDir, gesture_models, n_states)

## Adhoc hmm
if mode == 2:
    list_gesture = {("rectangle", n_states*4), ("triangle", n_states*3), ("caret", n_states*2), ("v", n_states*2), ("x", n_states*3),
            ("left_sq_bracket", n_states*3), ("right_sq_bracket", n_states*3), ("delete", n_states*4), ("star", n_states*4),
            ("arrow", n_states*4), ("check", n_states*2), ("circle", n_states*4), ("left_curly_brace", n_states*6),
            ("right_curly_brace", n_states*6), ("pigtail", n_states*4), ("question_mark", n_states*4)}
    #results = adhoc_test(gestureDir, list_gesture)

# Print results
print(results)