from dataset import *
from gesture import *

######### Gesture #########
# Creates leap motion unica dataset (Normalise and Sample)
def leapmotion_dataset(baseDir):
    # Crea file csv gesture
    # Rectangle
    DatasetIterator.create_gesture_dataset(baseDir, 'rectangle', 72)
    # Triangle
    DatasetIterator.create_gesture_dataset(baseDir, 'triangle', 54)
    # Caret
    DatasetIterator.create_gesture_dataset(baseDir, 'caret', 36)
    # V
    DatasetIterator.create_gesture_dataset(baseDir, 'v', 36)
    # X
    DatasetIterator.create_gesture_dataset(baseDir, 'x', 54)
    # Square Bracket Left
    DatasetIterator.create_gesture_dataset(baseDir, 'square-braket-left', 54)
    # Square Bracket Right
    DatasetIterator.create_gesture_dataset(baseDir, 'square-braket-right', 54)
    # Delete
    DatasetIterator.create_gesture_dataset(baseDir, 'delete', 54)
    #  Star
    DatasetIterator.create_gesture_dataset(baseDir, 'star', 90)
    # Left
    DatasetIterator.create_gesture_dataset(baseDir, 'left', 20)
    # Right
    DatasetIterator.create_gesture_dataset(baseDir, 'right', 20)

    return

# Makes one dollar dataset (Normalise and Sample)
def onedollar_dataset(baseDir):
    # Arrow
    DatasetIterator.create_gesture_dataset(baseDir, 'arrow/', 72)
    # Caret
    DatasetIterator.create_gesture_dataset(baseDir, 'caret/', 36)
    # Check
    DatasetIterator.create_gesture_dataset(baseDir, 'check/', 36)
    # Circle
    DatasetIterator.create_gesture_dataset(baseDir, 'circle/', 72)
    # Delete
    DatasetIterator.create_gesture_dataset(baseDir, 'delete_mark/', 54)
    # Left curly brace
    DatasetIterator.create_gesture_dataset(baseDir, 'left_curly_brace/', 36)
    # Right curly brace
    DatasetIterator.create_gesture_dataset(baseDir, 'right_curly_brace/', 36)
    # Left square bracket
    DatasetIterator.create_gesture_dataset(baseDir, 'left_sq_bracket/', 54)
    # Right square bracket
    DatasetIterator.create_gesture_dataset(baseDir, 'right_sq_bracket/', 54)
    # Pigtail
    DatasetIterator.create_gesture_dataset(baseDir, 'pigtail/', 36)
    # Question mark
    DatasetIterator.create_gesture_dataset(baseDir, 'question_mark/', 90)
    # Rectangle
    DatasetIterator.create_gesture_dataset(baseDir, 'rectangle/', 72)
    # Star
    DatasetIterator.create_gesture_dataset(baseDir, 'star/', 90)
    # Triangle
    DatasetIterator.create_gesture_dataset(baseDir, 'triangle/', 54)
    # V
    DatasetIterator.create_gesture_dataset(baseDir, 'v/', 36)
    # X
    DatasetIterator.create_gesture_dataset(baseDir, 'x/', 54)

######### Primitive #########
# Makes primitive dataset (from right and left movements)
def primitive_dataset(datasetDir, baseDir):
    # Up
    make_primitive_dataset(datasetDir, baseDir, Primitive.up)
    # Down
    make_primitive_dataset(datasetDir, baseDir, Primitive.down)
    # Diagonal
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, -145) # - 145
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, -135) # - 135
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, -130) # - 130
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, -60) # - 60
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, -45) # - 45
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, 135) # 135
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, 60) # 60
    make_primitive_dataset(datasetDir, baseDir, Primitive.diagonal, 45) # 45
    # Forward
    make_primitive_dataset(datasetDir, baseDir, Primitive.forward)
    # Behind
    make_primitive_dataset(datasetDir, baseDir, Primitive.behind)

    return

datasetDir = '/home/alessandro/Scaricati/gestures/'
baseDir  = '/Users/davide/Google Drive/Dottorato/Software/python/hmmtest/repository/'

mode = 4

if mode == 1:
    inputDir = baseDir + 'original/1dollar-dataset/'
    outputDir = baseDir + 'deictic/1dollar-dataset/raw'
    converter = Dollar1Converter()
    converter.create_deictic_dataset(inputDir, outputDir)

if mode == 2:
    inputDir = baseDir + 'original/unica-dataset/'
    outputDir = baseDir + 'deictic/unica-dataset/raw'
    converter = UnicaConverter()
    converter.create_deictic_dataset(inputDir, outputDir)

if mode == 3:
    inputDir = baseDir + 'deictic/unica-dataset/raw/left/'
    outputDir = baseDir + 'deictic/unica-dataset/scaled/left/'
    dataset = CsvDataset(inputDir)
    transform1 = ScaleDatasetTransform(scale=1)
    transform2 = NormaliseLengthTransform(axisMode=False)
    transform3 = CenteringTransform()
    dataset.addTransform(transform2)
    #dataset.addTransform(transform1)
    dataset.addTransform(transform3)
    dataset.applyTransforms(outputDir)


    dataset = CsvDataset(outputDir)
    dataset.plot()

if mode == 4:
    inputDir = baseDir + 'deictic/1dollar-dataset/raw/rectangle/'
    outputDir = baseDir + 'deictic/1dollar-dataset/resampled/rectangle/'
    dataset = CsvDataset(inputDir)

    transform1 = ResampleInSpaceTransform()

    dataset.plot(sampleName='1_fast_rectangle_01.csv')
    dataset.addTransform(transform1)
    transform2 = NormaliseLengthTransform(axisMode=False)
    transform3 = CenteringTransform()
    #dataset.addTransform(transform2)
    #dataset.addTransform(transform3)
    dataset.applyTransforms(outputDir)

    dataset = CsvDataset(outputDir)
    dataset.plot(sampleName='1_fast_rectangle_01.csv')





