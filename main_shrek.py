from dataset import CsvDatasetExtended
import matplotlib.pyplot as plt
import numpy
import csv
from os import listdir, rename
from os.path import isfile, join
# ground terms #
from model import *
from test import Test
from gesture import CreateRecognizer, CreateRecognizer

dir = '/home/ale/Scrivania/'
dir_original = dir+'original/'
dir_raw = dir+'raw/'
dir_resampled = dir+'resampled/'
num_states = 10
spu = 20

# gesture labels in training dataset
gesture_labels={
    '^':'caret',
    '[]':'square',
    'O':'circle',
    'X':'x',
    'V':'v'
}
# num of primitives for each gesture
gesture_primitives={
    'square': 4,
    'circle': 4,
    'x': 3,
    'v': 2,
    'caret': 2,
}
# deictic expression for each gesture
gesture_expressions={
    'square':[Point(0,0)+Line(-3,0)+Line(0,-3)+Line(3,0)+Line(0,3)],
    'v':[Point(0,0)+Line(-2,-2)+Line(-2,2)],
    'x':[Point(0,0)+Line(-2,-2)+Line(0,3)+Line(2,-2)],
    'circle':[Point(0,0)+Arc(-3,-3,cw=False)+Arc(3,-3,cw=False)+Arc(3,3,cw=False)+Arc(-3,3,cw=False),
              Point(0,0)+Arc(-3,3)+Arc(3,3)+Arc(3,-3)+Arc(-3,-3)],
    'caret':[Point(0,0)+Line(-2,2)+Line(-2,-2)]#,[Point(0,0)+Line(2,2)+Line(2,-2)]
}
# test dataset for each gesture
gesture_datasets={
    'square': [CsvDatasetExtended(dir=dir_resampled+'square/')],
    'circle': [CsvDatasetExtended(dir=dir_resampled+'circle/')],
    'x': [CsvDatasetExtended(dir=dir_resampled + 'x/')],
    'caret': [CsvDatasetExtended(dir=dir_resampled+'caret/')],
    'v': [CsvDatasetExtended(dir=dir_resampled+'v/')]
}

def convert_txt_csv():
    original_files = [f for f in listdir(dir_original) if isfile(join(dir_original, f))]
    for f in original_files:
        filename = f.split('.')[0]
        rename(dir_original+f, dir_original+filename+'.csv')

def plot_convert(dataset_path=dir_original, plot=False):
    training_samples = CsvDatasetExtended(dataset_path, type=str).readDataset()
    for file in training_samples:
        points = file.getPoints(columns=[0,44,45,-1])
        # take points
        p = []
        is_start = False
        is_end = False
        for point in points:
            # check if is end
            if point[0] == '-1':
                if is_start:
                    is_end = True
            # add point
            if is_start==True and is_end==False:
                p.append([float(point[1]),float(point[2])])
            # check if is start
            if point[0] == '-1':
                if not is_start:
                    is_start = True
                    gesture = point[-1]
        # plot points
        if plot:
            print(gesture)
            points = numpy.asarray(p)
            plt.scatter(points[:, 0], points[:, 1])
            for i in range(0, len(points)):
                plt.annotate(str(i), (points[i, 0], points[i, 1]))
            plt.plot(points[:, 0], points[:, 1], label=file.filename)
            plt.show()
        # save points
        with open(dir_raw+gesture_labels[gesture]+'/'+file.filename, "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(p)

def plot_dataset(label, folder=dir_resampled, compare_with_model=False):
    dataset = CsvDatasetExtended(dir=dir_resampled+label+'/')
    if compare_with_model:
       model = CreateRecognizer.createHMM(expression=gesture_expressions[label][0], num_states=num_states * gesture_primitives[label], sdu=spu)
       dataset.plot(compared_model=model)
    else:
        dataset.plot()

def normalize(directories, num_samples=20):
    for label in directories:
        # CsvDataset
        dataset = CsvDatasetExtended(dir=dir+label+'/')
        # Transform
        transform1 = NormaliseLengthTransform(axisMode=True)
        transform2 = ScaleDatasetTransform(scale=100)
        transform3 = CenteringTransform()
        print(num_samples*gesture_primitives[label])
        transform4 = ResampleInSpaceTransform(samples=num_samples * gesture_primitives[label])
        dataset.addTransforms(transforms=[transform1, transform2, transform3, transform4])
        dataset.applyTransforms(dir_resampled+label+'/')

def try_models(expressions=[], num_samples=10):
    # check
    if not isinstance(expressions, list):
        raise TypeError("expressions must be an array")

    for expression in expressions:
        # generate model
        model = CreateRecognizer.createHMM(expression)
        for i in range(num_samples):
            # generate sequence
            sequence = numpy.array(model.sample()).astype('float')
            # plot sequence
            plt.scatter(sequence[:, 0], sequence[:, 1])
            for i in range(0, len(sequence)):
                plt.annotate(str(i), (sequence[i, 0], sequence[i, 1]))
            plt.plot(sequence[:, 0], sequence[:, 1], label='Generated sample n°'+str(i))
            plt.show()

def test_recognition():
    # generate models
    models = CreateRecognizer.generateHMMs(expressions=gesture_expressions, num_states=6, spu=20)
    # start log-probability-based test (Test will create the gesture hmms from gesture_expressions)
    results = Test.offlineTest(gesture_hmms=models,
                               gesture_datasets=gesture_datasets)
    # show result through confusion matrix
    results.confusion_matrix.plot()


    models_2 = CreateRecognizer.createHMMs(expressions=gesture_expressions, num_states=6, spu=20)
    results = Test.offlineTest(gesture_hmms=models_2,
                               gesture_datasets=gesture_datasets)
    results.confusion_matrix.plot()

#plot_convert()
#plot_dataset(label='circle', compare_with_model=True)
#try_models(expressions=[Point(0,0)+Line(-2,2)+Line(-2,-2)])
#normalize(['caret','circle','square','v','x'])
test_recognition()