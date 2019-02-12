from dataset import CsvDatasetExtended
import matplotlib.pyplot as plt
import numpy
import csv
from os import listdir, rename
from os.path import isfile, join
# ground terms #
from model import *
from test import Test
from gesture import ModelExpression

dir = '/home/ale/Scrivania/'
dir_original = dir+'original/'
dir_raw = dir+'raw/'
dir_resampled = dir+'resampled/'
gesture_labels={
    '^':'caret',
    '[]':'square',
    'O':'circle',
    'X':'x',
    'V':'v'
}
primitive_options={
    'square': 4,
    'circle': 4,
    'v': 2,
    'caret': 2,
    'x': 3
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

def plot_dataset(label, folder=dir_resampled):
   dataset = CsvDatasetExtended(dir=dir_resampled+label+'/')
   for file in dataset.readDataset():
       file.plot()


def normalize(directories, num_samples=20):
    for label in directories:
        # CsvDataset
        dataset = CsvDatasetExtended(dir=dir+label+'/')
        # Transform
        transform1 = NormaliseLengthTransform(axisMode=True)
        transform2 = ScaleDatasetTransform(scale=100)
        transform3 = CenteringTransform()
        print(num_samples*primitive_options[label])
        transform4 = ResampleInSpaceTransform(samples=num_samples * primitive_options[label])
        dataset.addTransforms(transforms=[transform1, transform2, transform3, transform4])
        dataset.applyTransforms(dir_resampled+label+'/')

def try_models(expressions=[]):
    for expression in expressions:
        model = ModelExpression.createHmm(expression)


def track_gesture():
    expressions={
        #'square':[Point(0,0)+Line(-3,0)+Line(0,-3)+Line(3,0)+Line(0,3), Point(0,0)+Line(0,3)+Line(-3,0)+Line(0,-3)+Line(3,0)],
        'v':[Point(0,0)+Line(-2,-2)+Line(-2,2)],
        #'x':[Point(0,0)+Line(-2,-2)+Line(0,3)+Line(2,2)],
        #'circle':[Point(0,0)+Arc(-3,-3,cw=False)+Arc(3,-3,cw=False)+Arc(3,3,cw=False),Arc(-3,3,cw=False)],
                  #[Point(0,0)+Arc(-3,3)+Arc(3,3)+Arc(3,-3)+Arc(-3,-3)]],
        'caret':[Point(0,0)+Line(-2,2)+Line(-2,-2)]#,[Point(0,0)+Line(2,2)+Line(2,-2)]
    }
    gesture_expressions = {}
    for k,v in expressions.items():
        gesture_expressions[k] = []
        for expression in v:
            gesture_expressions[k].append(ModelExpression.createHmm(expression))

    gesture_dataset = {
        #'square': [CsvDatasetExtended(dir_resampled+'square/', type=float)],
        'v': [CsvDatasetExtended(dir_resampled + "v/", type=float)],
        #'x': [CsvDatasetExtended(dir_resampled + "x/")],
        #'circle': [CsvDataset(dir + "circle/")],
        'caret': [CsvDatasetExtended(dir + "caret/")],
    }

    # start log-probability-based test (Test will create the gesture hmms from gesture_expressions)
    results = Test.offlineTest(gesture_hmms=gesture_expressions,
                               gesture_datasets=gesture_dataset)
    # show result through confusion matrix
    results.confusion_matrix.plot()
    # save result on csv file
    #results.save(path=None)
    # analyse wrong classifications

#plot_convert()
plot_dataset(label='caret')
#normalize(['caret','circle','square','v','x'])
#track_gesture()