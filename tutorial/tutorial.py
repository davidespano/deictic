from gesture import ModelFactory, DatasetExpressions
from model.gestureModel import Point, Line, Arc
from dataset import CsvDataset
from test import Test
from config import Config

def firstExample():
    '''
        shows how to describe a gesture with Deictic and create its model.
    '''
    # describe gesture swipe_right through deictic
    gesture_expressions = {
            'swipe_right':
                [
                    Point(0,0)+Line(3,0)
                ]
            }
    # create deictic model for swipe_right
    hmm_swipe_right = ModelFactory.createHmm(gesture_expressions = gesture_expressions, num_states = 6, num_samples = 20)
    print(hmm_swipe_right)


def secondExample():
    '''
        secondExample, starting from firstExample, shows how to define a set of CsvDataset, how to compare two different gestures and, finally, how to plot the results.
    '''
    # describe gesture swipe_right through deictic
    gesture_expressions = {
        'swipe_right':
            [
                Point(0, 0) + Line(3, 0)
            ],
        'swipe_left':
            [
                Point(3, 0) + Line(-3, 0)
            ]
    }
    # get swipe right and swipe left datasets
    gesture_dataset = None
    # create deictic model for swipe right and left
    gesture_hmms = ModelFactory.createHmm(gesture_expressions)
    # start log-probability-based test
    results = Test.getInstance().offlineTest(gesture_hmms=gesture_hmms, gesture_datasets=gesture_dataset)
    # show result through confusion matrix
    results.plot()
    # save result on csv file
    results.save(path=None)


def thirdExample():
    '''
        thirdExample shows how to compare the gestures described in 1$ multistroke dataset (http://depts.washington.edu/madlab/proj/dollar/ndollar.html ).
        In DatasetExpressions you can find the gesture descriptions of different datasets:
        in addition to 1$ multistroke, 1$ unistroke dataset (http://depts.washington.edu/madlab/proj/dollar/index.html),
        leap motion unica dataset (from the University of Cagliari and created by prof. Lucio Davide Spano) and Shrec2017(http://www-rech.telecom-lille.fr/shrec2017-hand/).
    '''
    # get the gesture expressions which describe 1$ unistroke dataset
    gesture_expressions = DatasetExpressions.returnExpressions(selected_dataset= DatasetExpressions.TypeDataset.multistroke_1dollar)
    # get gesture datasets
    gesture_dataset = {
        'arrowhead' :  [CsvDataset(Config.baseDir+"deictic/mdollar-dataset/resampled/arrowhead/")],
        'asterisk': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/asterisk/")],
        'D': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/D/")],
        'exclamation_point': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/exclamation_point/")],
        'H': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
        'half_note': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
        'I': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
        'line': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
        'N': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
        'null': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
        'P': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
        'pitchfork': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
        'six_point_star': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
        'T': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
        'X': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
    }

    # start log-probability-based test (Test will create the gesture hmms from gesture_expressions)
    results = Test.getInstance().offlineTestExpression(gesture_expressions=gesture_expressions, gesture_datasets=gesture_dataset)
    # show result through confusion matrix
    results.plot()
    # save result on csv file
    results.save(path=None)




thirdExample()