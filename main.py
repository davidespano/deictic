from dataset import *
from gesture import *
from test import *


# Get_SubDirectories
# Get all subdirectories from the choosen directory
@staticmethod
def get_subdirectories(baseDir):
    return [name for name in os.listdir(baseDir)
            if os.path.isdir(os.path.join(baseDir, name))]

# Test compare gesture
def compare_gesture(baseDir, results, n_states, index = 1):

    baseDir = baseDir + 'normalised-trajectory/'
    #### Test ####
    models = []
    models.append(create_hmm_gesture(baseDir, 'caret', n_states*2, index))
    #models.append(create_hmm_gesture('delete', baseDir, n_states*3, index))
    #models.append(create_hmm_gesture('left', baseDir, n_states, index))
    models.append(create_hmm_gesture(baseDir, 'rectangle', n_states*4, index))
    #models.append(create_hmm_gesture('right', baseDir, n_states, index))
    models.append(create_hmm_gesture(baseDir,'square-braket-left', n_states*3, index))
    models.append(create_hmm_gesture(baseDir,'square-braket-right', n_states*3, index))
    models.append(create_hmm_gesture(baseDir,'triangle', n_states*3, index))
    models.append(create_hmm_gesture(baseDir,'v', n_states*2, index))
    models.append(create_hmm_gesture(baseDir,'x', n_states*3, index))

    return compare_all_models_test_without_primitive(models, baseDir, results, index = index)

# Test Compare Modelled Gesture
def compare_modelled_gestures(primitiveDir, gestureDir, n_states):
    #### Test ####
    models = []
    # Caret
    caret, seq = create_caret(primitiveDir, n_states)
    models.append(caret)
    # Delete
    delete, seq = create_delete(primitiveDir, n_states)
    models.append(delete)
    # Left
    left = primitive_model(primitiveDir, n_states, direction = Primitive.left)
    models.append(left)
    # Rectangle
    rectangle, seq = create_rectangle(primitiveDir, n_states)
    models.append(rectangle)
    # Right
    right = primitive_model(primitiveDir, n_states, direction = Primitive.right)
    models.append(right)
    # Square Braket Left
    sq_braket_left, seq = create_square_braket_left(primitiveDir, n_states)
    models.append(sq_braket_left)
    # Square Braket Right
    sq_braket_right, seq = create_square_braket_right(primitiveDir, n_states)
    models.append(sq_braket_right)
    # Triangle
    triangle, seq = create_triangle(primitiveDir, n_states)
    models.append(triangle)
    # V
    v, seq = create_v(primitiveDir, n_states)
    models.append(v)
    # X
    #x, seq = create_x(primitiveDir, n_states)
    #models.append(x)

    #for model in models:
    #    print(model.name)
    #    plot_gesture(model)

    # Compare
    return compare_all_models_test(models, gestureDir+'down-trajectory/')

baseDir = '/home/alessandro/Scaricati/dataset/leap_motion_unica/'
primitiveDir = '/home/alessandro/Scaricati/dataset/leap_motion_unica/'
#results = numpy.zeros((8, 8), dtype=numpy.int)
results = compare_modelled_gestures(primitiveDir, baseDir, n_states=8)
print(results)