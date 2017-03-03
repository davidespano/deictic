from dataset import *
from gesture import *

def dataset_factory(list, inputDir, outputDir, unistroke_mode = True):

    for gesture in list:
        input_dir = inputDir+gesture[0]+'/'
        output_dir = outputDir+gesture[0]+'/'
        dataset = CsvDataset(input_dir)

        # Transform
        if unistroke_mode:
            transform1 = NormaliseLengthTransform(axisMode=True)
        else:
            transform1 = NormaliseLengthTransform(axisMode=gesture[3])
        transform2 = ScaleDatasetTransform(scale=100)
        transform3 = CenteringTransform()
        #transform4 = RotateCenterTransform(traslationMode=True)
        if unistroke_mode:
            transform5 = ResampleInSpaceTransform(samples=gesture[1])
        else:
            transform5 = ResampleInSpaceTransformMultiStroke(samples=gesture[1], strokes = gesture[2])
        # Apply transforms
        dataset.addTransform(transform1)
        dataset.addTransform(transform2)
        dataset.addTransform(transform3)
        #dataset.addTransform(transform4)
        dataset.addTransform(transform5)

        dataset.applyTransforms(output_dir)
    return


#baseDir = '/home/alessandro/PycharmProjects/deictic/repository/'
baseDir  = '/Users/davide/Google Drive/Dottorato/Software/python/hmmtest/repository/'

mode = 3
n_sample = 40

# Unica
if mode == 1:
    list = {("rectangle", 4*n_sample), ("triangle", 3*n_sample), ("caret", 2*n_sample), ("v", 2*n_sample), ("x", 3*n_sample),
            ("square-braket-left", 3*n_sample), ("square-braket-right", 3*n_sample), ("delete", 3*n_sample),
            ("star", 5*n_sample)
            }#, ("left", n_sample), ("right", n_sample)
    dataset_factory(list, baseDir+'deictic/unica-dataset/raw/', baseDir+'deictic/unica-dataset/resampled/')

# 1Dollar
if mode == 2:
    # name and n_samples
    list = {("arrow", 4*n_sample), ("caret", 2*n_sample), ("circle", 4*n_sample), ("check", 2*n_sample), ("delete_mark", 3*n_sample),
            ("left_curly_brace", 6*n_sample), ("left_sq_bracket", 3*n_sample), ("pigtail", 4*n_sample), ("question_mark", 4*n_sample),
            ("rectangle", 4*n_sample), ("right_curly_brace", 6*n_sample), ("right_sq_bracket", 3*n_sample), ("star", 5*n_sample),
            ("triangle", 3*n_sample), ("v", 2*n_sample), ("x", 3*n_sample)
            }
    dataset_factory(list, baseDir+'deictic/1dollar-dataset/raw/', baseDir+'deictic/1dollar-dataset/resampled/')

# MDollar
if mode == 3:
    # Name gesture, n samples and strokes
    list = {("arrowhead", n_sample, 2, True), ("asterisk", n_sample, 3, True), ("D", n_sample, 2, True), ("exclamation_point", n_sample, 2, False),
            ("five_point_star", n_sample, 1, True), ("H", n_sample, 3, True), ("half_note", n_sample, 2, True),
            ("I", n_sample, 3, True), ("N", n_sample, 3, True), ("null", n_sample, 2, True),
            ("P", n_sample, 2, True), ("pitchfork", n_sample, 2, True), ("six_point_star", n_sample, 2, True),
            ("T", n_sample, 2, True), ("X", n_sample, 2, True)
            }#("line", n_sample, 1)
    dataset_factory(list, baseDir+'deictic/mdollar-dataset/raw/', baseDir+'deictic/mdollar-dataset/resampled/', unistroke_mode=False)

## Original
# Unica
if mode == 10:
    inputDir = baseDir + 'original/unica-dataset/'
    outputDir = baseDir + 'deictic/unica-dataset/raw'
    converter = UnicaConverter()
    converter.create_deictic_dataset(inputDir, outputDir)
# 1Dollar
if mode == 11:
    inputDir = baseDir + 'original/1dollar-dataset/'
    outputDir = baseDir + 'deictic/1dollar-dataset/raw'
    converter = Dollar1Converter()
    converter.create_deictic_dataset(inputDir, outputDir)
# MDollar
if mode == 12:
    inputDir = baseDir + 'original/mdollar-dataset'
    outputDir = baseDir + 'deictic/mdollar-dataset/raw'
    converter = DollarMConverter()
    converter.order_files(inputDir, inputDir)
    converter.create_deictic_dataset(inputDir, outputDir)


# Prove
if mode == 13:
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

if mode == 14:
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

if mode == 15:
    inputDir = baseDir + 'deictic/1dollar-dataset/raw/left_curly_brace/'
    dataset = CsvDataset(inputDir)
    dataset.plot(singleMode=True)

# Diagonal
if mode == 16:
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
if mode == 18:
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






