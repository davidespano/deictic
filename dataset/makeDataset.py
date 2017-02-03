from gesture import *
from dataset import *
import os

## make_gesture_dataset
# Makes dataset (original, normalize and down-trajectory) for the input gesture
def make_gesture_dataset(baseDir, gestureName, type = TypeFile.csv,  sample = 20):
    # Creates Folders
    ToolsDataset.create_folder(baseDir, gestureName)

    # Makes File
    if(type == TypeFile.csv):# Csv files
        # Original
        ToolsDataset.find_gesture_file('/home/alessandro/Scaricati/gestures/', baseDir, gestureName)
        ToolsDataset.replace_csv(baseDir+'original/'+ gestureName +'/')
    else: # Xml files
        ToolsDataset.xml_to_csv('/home/alessandro/Scaricati/xml', gestureName, baseDir)
    dataset = ToolsDataset(baseDir + 'original/' + gestureName + '/')

    # Normalised
    dataset.normalise(baseDir + 'normalised-trajectory/' + gestureName +'/')
    dataset = ToolsDataset(baseDir + 'normalised-trajectory/' + gestureName+'/')
    # Down Samples
    dataset.down_sample(baseDir + 'down-trajectory/' + gestureName+'/', sample)

    return

## make_primitive_dataset
# Makes dataset (original, normalize and down-trajectory) for the input primitive
def make_primitive_dataset(datasetDir, baseDir, direction, degree=0, sample=20):
    ### 2D
    # Left
    if direction == Primitive.left:
        dataset = NormaliseSamples(datasetDir + '/left/')
        name = 'left'
    # Right
    elif direction == Primitive.right:
        dataset = NormaliseSamples(datasetDir + '/right/')
        name = 'right'
    # Up
    elif direction == Primitive.up:
        dataset = NormaliseSamples(datasetDir + '/right/')
        name = 'up'
    # Down
    elif direction == Primitive.down:
        dataset = NormaliseSamples(datasetDir + '/left/')
        name = 'down'
    # Diagonal
    elif direction == Primitive.diagonal:
        dataset = NormaliseSamples(datasetDir + '/right/')
        name = 'diagonal_{}'.format(degree)

    ### 3D
    # Forward
    elif direction == Primitive.forward :
        dataset = NormaliseSamples(datasetDir + '/right/')
        name = 'forward'
    # Behind
    elif direction == Primitive.behind:
        dataset = NormaliseSamples(datasetDir + '/left/')
        name = 'behind'

    # Create folder if not exist
    if not os.path.exists(datasetDir + '/' + name):
        os.makedirs(datasetDir + '/' + name)

    # Makes 'original' dataset
    # Up or Down
    if direction == Primitive.up or direction == Primitive.down:
        # Swap
        dataset.swap(datasetDir + '/' + name, name)
    # Diagonal
    elif direction == Primitive.diagonal:
        # Rotate
        dataset.rotate_lines(datasetDir + '/' + name + '/', name, degree)
    # Forward or Behind
    elif direction == Primitive.forward or direction == Primitive.behind:
        # Swap z
        dataset.swap(datasetDir + '/' + name + '/', name, dimensions = 3)

    # Normalised and Samples
    ToolsDataset.make_gesture_dataset(baseDir, name)

    return