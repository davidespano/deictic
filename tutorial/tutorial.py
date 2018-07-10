from gesture import ModelExpression, DatasetExpressions, DatasetFolders, TypeDataset
from model.gestureModel import Point, Line, Arc
from dataset import *#CsvDataset, CsvDatasetExtended, Sequence, ResampleInSpaceTransformOnline
from test import Test
from config import Config
from real_time.tree_test import *

import numpy


def firstExample():
    '''
        shows how to describe a gesture with Deictic and create its model.
    '''
    # describe gesture swipe_right through deictic
    expressions = {
            'swipe_right':
                [
                    Point(0,0)+Line(3,0)
                ]
            }
    # create deictic model for swipe_right
    hmm_swipe_right = ModelExpression.generatedModels(expressions = expressions, num_states = 6, spu = 20)
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
    gesture_hmms = ModelExpression.generatedModels(gesture_expressions)
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
    # get the gesture expressions which describe 1$ multistroke dataset
    gesture_expressions = DatasetExpressions.returnExpressions(selected_dataset= DatasetExpressions.TypeDataset.multistroke_1dollar)
    # get gesture datasets
    gesture_dataset = {
        'arrowhead' :  [CsvDataset(Config.baseDir+"deictic/mdollar-dataset/resampled/arrowhead/")],
        'asterisk': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/asterisk/")],
        'D': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/D/")],
        'exclamation_point': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/exclamation_point/")],
        'H': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
        'half_note': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/half_note/")],
        'I': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/I/")],
        'N': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/N/")],
        'null': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/null/")],
        'P': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/P/")],
        'pitchfork': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/pitchfork/")],
        'six_point_star': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/six_point_star/")],
        'T': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/T/")],
        'X': [CsvDataset(Config.baseDir + "deictic/mdollar-dataset/resampled/X/")],
    }

    # start log-probability-based test (Test will create the gesture hmms from gesture_expressions)
    results = Test.getInstance().offlineTestExpression(gesture_expressions=gesture_expressions, gesture_datasets=gesture_dataset)
    # show result through confusion matrix
    results.plot()
    # save result on csv file
    results.save(path=None)

def fourthExample():
    '''

    :return:
    '''
    # get the gesture expressions which describe 1$ unistroke dataset
    gesture_expressions = DatasetExpressions.returnExpressions(selected_dataset=DatasetExpressions.TypeDataset.unistroke_1dollar)
    # create hmms
    gesture_hmms = ModelExpression.generatedModels(expressions=gesture_expressions)
    # get sequence test (by using the first model of circle for generating a sample)
    sequence_test = gesture_hmms['circle'][0].sample()
    # compare hmms and show the computed log probabilities for each gesture
    index_label, log_probabilities = Test.compare(sequence=sequence_test, gesture_hmms=gesture_hmms, return_log_probabilities=True)
    print("The gesture with the highest log probabilities value is " +index_label)
    print(log_probabilities)

def fifthExample():
    """
        online
    :return:
    """
    # gesture expressions
    expressions = {
        'arrow': [Point(0,0)  +  Line(6,4)  +  Line(-4,0)  +  Line(5,1)  +  Line(-1,-4)],
        'caret': [Point(0,0)  +  Line(2,3)  +  Line(2,-3)],
        'check': [Point(0,0)  +  Line(2,-2)  +  Line(4,6)],
        'circle': [Point(0,0)  +  Arc(-3,-3,False)  +  Arc(3,-3,False)  +  Arc(3,3,False)  +  Arc(-3,3,False)],
        'delete_mark': [Point(0,0)  +  Line(2,-3)  +  Line(-2,0)  +  Line(2,3)],
        #'left_curly_brace': [Point(0,0)  +  Arc(-5,-5,False)  +  Arc(-3,-3,True)  +  Arc(3,-3,True)  +  Arc(5,-5,False)],
        'left_sq_bracket': [Point(0,0)  +  Line(-4,0)  +  Line(0,-5)  +  Line(4,0)],
        #'pigtail': [Point(0,0)  +  Arc(3,3,False)  +  Arc(-1,1,False)  +  Arc(-1,-1,False)  +  Arc(3,-3,False)],
        #'question_mark': [Point(0,0)  +  Arc(4,4,True)  +  Arc(4,-4,True)  +  Arc(-4,-4,True)  +  Arc(-2,-2,False)  +  Arc(2,-2,False)],
        'rectangle': [Point(0,0)  +  Line(0,-3)  +  Line(4,0)  +  Line(0,3)  +  Line(-4,0)],
        #'right_curly_brace': [Point(0,0)  +  Arc(5,-5,True)  +  Arc(3,-3,False)  +  Arc(-3,-3,False)  +  Arc(-5,-5,True)],
        'right_sq_bracket': [Point(0,0)  +  Line(4,0)  +  Line(0,-5)  +  Line(-4,0)],
        'star': [Point(0,0)  +  Line(2,5)  +  Line(2,-5)  +  Line(-5,3)  +  Line(6,0)  +  Line(-5,-3)],
        'triangle': [Point(0,0)  +  Line(-3,-4)  +  Line(6,0)  +  Line(-3,4)],
        'v': [Point(0,0)  +  Line(2,-3)  +  Line(2,3)],
        'x': [Point(0,0)  +  Line(3,-3)  +  Line(0,3)  +  Line(-3,-3)]
    }
    # datasets
    gesture_dataset = {
        'arrow': (4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/arrow/")]),
        'caret': (2,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/caret/")]),
        'check': (2,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/check/")]),
        'circle':(4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/circle/")]),
        'delete_mark': (3,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/delete_mark/")]),
        #'left_curly_brace': (6,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/left_curly_brace/")]),
        'left_sq_bracket': (3,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/left_sq_bracket/")]),
        #'pigtail': (4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/pigtail/")]),
        #'question_mark': (4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/question_mark/")]),
        'rectangle': (4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/rectangle/")]),
        #'right_curly_brace': (6,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/right_curly_brace/")]),
        'right_sq_bracket': (3,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/right_sq_bracket/")]),
        'star': (5,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/star/")]),
        'triangle': (3, [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/triangle/")]),
        'v': (2,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/v/")]),
        'x': (3,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/online/x/")])
    }

    # Tree
    tree=Tree(gesture_exp=expressions)
    # start log-probability-based test (Test will create the gesture hmms from gesture_expressions)
    results = Test.getInstance().onlineTest(tree=tree,
                                           gesture_datasets=gesture_dataset,
                                           perc_completed=100)
    # show result through confusion matrix
    results.plot()
    # save result on csv file
    #results.save(path=None)

# Start example
fifthExample()