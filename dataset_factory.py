from dataset import *
from gesture import *

mode = 4

######### Gesture #########
# Creates unica dataset
def dataset_gesture_factory(list, inputDir, outputDir):

    for gesture in list:
        input_dir = inputDir+gesture[0]+'/'
        output_dir = outputDir+gesture[0]+'/'
        dataset = CsvDataset(input_dir)

        # Transform
        transform1 = NormaliseLengthTransform(axisMode=True)
        transform2 = ScaleDatasetTransform(scale=100)
        transform3 = CenteringTransform()
        transform4 = ResampleInSpaceTransform(samples=gesture[1])
        # Apply transforms
        dataset.addTransform(transform1)
        dataset.addTransform(transform2)
        dataset.addTransform(transform3)
        dataset.addTransform(transform4)

        dataset.applyTransforms(output_dir)

        #dataset = CsvDataset(output_dir)
        #dataset.plot()
    return

######### Primitive #########
# Makes primitive dataset (from right movement)
def dataset_primitive_factory(list_gestures, inputDir, outputDir, samples=20):

    for primitive in list_gestures:
        output_dir = outputDir+(str(primitive))+'/'
        dataset = CsvDataset(inputDir)

        # Transforms
        #transform1 = ScaleDatasetTransform(scale=1)
        transform1 = RotateTransform(traslationMode=False, cols=[0,1], theta=primitive)
        transform2 = CenteringTransform()
        transform3 = NormaliseLengthTransform(axisMode=False)
        transform4 = ScaleDatasetTransform(scale=100)
        transform5 = ResampleInSpaceTransform(samples = samples)

        # Apply transforms
        dataset.addTransform(transform1)
        dataset.addTransform(transform2)
        dataset.addTransform(transform3)
        dataset.addTransform(transform4)
        dataset.addTransform(transform5)

        dataset.applyTransforms(output_dir)

        #dataset = CsvDataset(output_dir)
        #print(str(primitive))
        #dataset.plot()

    return


baseDir = '/home/alessandro/PycharmProjects/deictic/repository/'
#baseDir  = '/Users/davide/Google Drive/Dottorato/Software/python/hmmtest/repository/'

# Unica
list = {("rectangle", 80), ("triangle", 60), ("caret", 40), ("v",40), ("x", 60),
        ("square-braket-left", 60), ("square-braket-right", 60), ("delete", 60),
        ("star", 100)
        }#("left", 20), ("right", 20)
#dataset_gesture_factory(list, baseDir+'deictic/unica-dataset/raw/', baseDir+'deictic/unica-dataset/resampled/')

# 1Dollar
list = {("rectangle", 80), ("triangle",60), ("caret",40), ("v",40), ("x",60),
        ("left_sq_bracket",60), ("right_sq_bracket",60), ("delete_mark", 60), ("star", 100), ("arrow",80),
        ("check",40), ("circle", 80), ("left_curly_brace",120), ("right_curly_brace", 120),
        ("pigtail", 80), ("question_mark", 80)}
dataset_gesture_factory(list, baseDir+'deictic/1dollar-dataset/raw/', baseDir+'deictic/1dollar-dataset/resampled/')

# Primitive
list = {340, 320, 270, 240, 200, 150, 145, 140, 135, 130, 120, 110, 90, 70, 60, 45, 40, 20,
        -340, -320, -250, -240, -200, -145, -140, -135, -130, -120, -110, -90, -70, -60, -45, -40, -20}

#dataset_primitive_factory(list, baseDir+'deictic/unica-dataset/raw/right/', baseDir+'deictic/unica-dataset/resampled/')

mode = 0

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
    #
    #transform1 = ScaleDatasetTransform(scale=1)
    #transform2 = NormaliseLengthTransform(axisMode=False)
    #transform3 = CenteringTransform()
    transform1 = ResampleInSpaceTransform()
    #
    dataset.addTransform(transform1)
    #dataset.addTransform(transform2)
    #dataset.addTransform(transform3)
    dataset.applyTransforms(outputDir)

    dataset = CsvDataset(outputDir)
    dataset.plot()

if mode == 4:
    inputDir = baseDir + 'deictic/unica-dataset/raw/right/'
    outputDir = baseDir + 'deictic/unica-dataset/resampled/right/'
    dataset = CsvDataset(inputDir)

    transform1 = ResampleInSpaceTransform()

    dataset.addTransform(transform1)
    transform2 = NormaliseLengthTransform(axisMode=False)
    transform3 = CenteringTransform()
    #dataset.addTransform(transform2)
    #dataset.addTransform(transform3)
    dataset.applyTransforms(outputDir)

    dataset = CsvDataset(outputDir)
    dataset.plot(sampleName='claudio_right.csv')

if mode == 5:
    inputDir = baseDir + 'deictic/1dollar-dataset/raw/left_curly_brace/'
    dataset = CsvDataset(inputDir)
    dataset.plot(singleMode=True)

# Diagonal
if mode == 5:
    inputDir = baseDir + '/deictic/unica-dataset/raw/right/'
    outputDir = baseDir + '/deictic/unica-dataset/scaled/up/'
    dataset = CsvDataset(inputDir)
    # Transforms
    transform1 = ScaleDatasetTransform(scale=1)
    transform2 = NormaliseLengthTransform(axisMode=False)
    transform3 = CenteringTransform()
    trasnform4 = RotateTransform(traslationMode=False, cols=[0,1], theta=90)
    # Apply transforms
    dataset.addTransform(transform3)
    dataset.addTransform(trasnform4)
    dataset.addTransform(transform2)
    dataset.applyTransforms(outputDir)

    dataset = CsvDataset(outputDir)
    dataset.plot()

# Samples
if mode == 6:
    inputDir = baseDir + '/deictic/unica-dataset/raw/right/'
    outputDir = baseDir + '/deictic/unica-dataset/scaled/right/'
    dataset = CsvDataset(inputDir)
    # Transforms
    transform1 = ScaleDatasetTransform(scale=1)
    transform2 = NormaliseLengthTransform(axisMode=False)
    transform3 = CenteringTransform()
    transform4 = Sampling(scale=20)
    # Apply transforms
    dataset.addTransform(transform4)
    dataset.addTransform(transform2)
    dataset.addTransform(transform3)
    dataset.applyTransforms(outputDir)

    dataset = CsvDataset(outputDir)
    dataset.plot()






