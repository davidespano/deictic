#### Library ####
# Csvdataset
#from real_time.parsing_trajectory.csv_dataset import CsvDataset
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
# Kalman filter
from pykalman import KalmanFilter
from dataset import *
from dataset.csvDataset import CsvDataset
###
from model.gestureModel import TypeRecognizer
from gesture.datasetExpressions import DatasetExpressions
from gesture.modellingExpression import ModelExpression
from real_time.parsing_trajectory.model_factory import Model
from real_time.parsing_trajectory.trajectory_parsing import Parsing
# Test
from test.test import Test, Result
# Levenshtein
from real_time.parsing_trajectory.levenshtein_distance import LevenshteinDistance
import sys

debug = 4

if debug == 0:
    # get the gesture expressions which describe 1$ multistroke dataset
    gesture_expressions = DatasetExpressions.returnExpressions(selected_dataset=DatasetExpressions.TypeDataset.unistroke_1dollar)
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/parsed/"
    # get gesture datasets
    gesture_datasets = {
        # "arrow": [CsvDataset(base_dir + "arrow/", type=str)],
        # "caret": [CsvDataset(base_dir + "caret/", type=str)],
        #"check": [CsvDataset(base_dir+"check/", type=str)],
        "circle": [CsvDataset(base_dir+"circle/", type=str)],
        # "delete_mark": [CsvDataset(base_dir + "delete_mark/", type=str)],
        # "left_curly_brace": [CsvDataset(base_dir + "left_curly_brace/", type=str)],
        # "left_sq_bracket": [CsvDataset(base_dir + "left_sq_bracket/", type=str)],
        #"pigtail": [CsvDataset(base_dir + "pigtail/", type=str)],
        # "question_mark": [CsvDataset(base_dir + "question_mark/", type=str)],
        # "rectangle": [CsvDataset(base_dir + "rectangle/", type=str)],
        # "right_curly_brace": [CsvDataset(base_dir + "right_curly_brace/", type=str)],
        # "right_sq_bracket": [CsvDataset(base_dir + "right_sq_bracket/", type=str)],
        #  "star": [CsvDataset(base_dir + "star/", type=str)],
        # "triangle": [CsvDataset(base_dir + "triangle/", type=str)],
        "v": [CsvDataset(base_dir + "v/", type=str)],
        # "x": [CsvDataset(base_dir + "x/", type=str)]
    }
    # hmms
    gesture_hmms = ModelExpression.generatedModels(expressions=gesture_expressions, type=TypeRecognizer.online, num_states=4)
    ### Debug ###
    # for i in range(0, 10):
    #     sample = gesture_hmms["caret"][0].sample()
    #     sample2 = gesture_hmms["question_mark"][0].sample()
    #     print("test: "+str(i))
    #     print("caret:")
    #     print(sample)
    #     print("question_mark:")
    #     print(sample2)
    #     print("\n")
    ### ### ###
    ### Test ###
    # start log-probability-based test (Test will create the gesture hmms from gesture_expressions)
    results = Test.getInstance().offlineTest(gesture_hmms, gesture_datasets)
    # show result through confusion matrix
    results.plot()
    ### ### ###

if debug == 1:
    # datasets path
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/"
    input_dir = base_dir + "raw/"
    output_dir = base_dir + "parsed/"
    # datasets list
    directories = ["arrow", "caret", "circle", "check", "delete_mark", "left_curly_brace", "left_sq_bracket", "pigtail", "question_mark",
                   "rectangle", "right_curly_brace", "right_sq_bracket", "star", "triangle", "v", "x"]
    directories = ['v']

    for directory in directories:

        print("Start " + directory)
        # original kalman + resampled
        dataset_original = CsvDataset(input_dir+directory+"/")
        kalmanTransform = KalmanFilterTransform()

        parse = ParseSamples()
        dataset_original.addTransform(kalmanTransform)

        #resampledTransform = ResampleTransform
        #resampledTransform = ResampleTransform(delta=5)
        #dataset_original.addTransform(resampledTransform)

        dataset_original.addTransform(parse)
        sequence_original = dataset_original.applyTransforms(output_dir=(output_dir+directory+"/"))
        print("End " + directory)



if debug == 2:
    # Get dataset
    k_cross_validation = 10
    n_states = 2
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/parsed/"
    directories = {
        # "arrow": CsvDataset(base_dir + "arrow/", type=str),
        # "caret": CsvDataset(base_dir + "caret/", type=str),
        # "check": CsvDataset(base_dir+"check/", type=str),
        "circle": CsvDataset(base_dir+"circle/", type=str),
        # "delete_mark": CsvDataset(base_dir + "delete_mark/", type=str),
        # "left_curly_brace": CsvDataset(base_dir + "left_curly_brace/", type=str),
        # "left_sq_bracket": CsvDataset(base_dir + "left_sq_bracket/", type=str),
        # "pigtail": CsvDataset(base_dir + "pigtail/", type=str),
        # "question_mark": CsvDataset(base_dir + "question_mark/", type=str),
        # "rectangle": CsvDataset(base_dir + "rectangle/", type=str),
        # "right_curly_brace": CsvDataset(base_dir + "right_curly_brace/", type=str),
        # "right_sq_bracket": CsvDataset(base_dir + "right_sq_bracket/", type=str),
        # "star": CsvDataset(base_dir + "star/", type=str),
        # "triangle": CsvDataset(base_dir + "triangle/", type=str),
        "v": CsvDataset(base_dir + "v/", type=str),
        # "x": CsvDataset(base_dir + "x/", type=str)
    }

    sum = 0
    cont_result = None
    # Get train and test folds
    start_time = time.time()#### debug time
    for k in range(k_cross_validation):
        print("Iteration: "+str(k))
        gesture_hmms = {}
        gesture_datasets = {}
        for directory in directories.keys():
                # create hmm
                model = Model(n_states = n_states, n_features = 1, name = directory)
                # get train and test samples
                train_samples,test_samples = directories[directory].crossValidation(iteration=k)
                model.train(train_samples)
                gesture_datasets[directory] = test_samples
                # add hmm to dictionary
                gesture_hmms[directory] = [model.getModel()]
                # debug
                print("Label: " + str(directory) + " - Train :"+str(len(train_samples)) + " - Test: "+ str(len(test_samples)))

        result = Test.getInstance().offlineTest(gesture_hmms=gesture_hmms, gesture_datasets=gesture_datasets)
        if cont_result == None:
            cont_result = result
        else:
            cont_result = cont_result + result
        print("--- %s seconds ---" % (time.time() - start_time)) # debug time
        #result.save(path="/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/parsed/iteration_"+str(k)+"-states_"+str(n_states)+".csv")
        sum += result.meanAccuracy()
    print("\nMean Accuracy: "+str(sum/k_cross_validation))
    cont_result.save(path="/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/parsed/" + "states_" + str(n_states) + ".csv")





######################## Debug
if debug == 3:
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/raw/"
    directories = ["circle"]
    files =[]#['1_medium_left_sq_bracket_08.csv']
    dataset_kalman_resampled = {}

    # Raw data with kalman and resampling
    for directory in directories:
        # original kalman + resampled
        dataset_original = CsvDataset(base_dir+directory+"/")

        kalmanTransform = KalmanFilterTransform()
        dataset_original.addTransform(kalmanTransform)

        #resampledTransform = ResampleInSpaceTransform(samples=40)
        resampledTransform = ResampleTransform(delta=10)
        dataset_original.addTransform(resampledTransform)

        dataset_kalman_resampled[directory] = dataset_original.applyTransforms()

    for index in range(len(dataset_kalman_resampled[directory])):
        data_plots = []
        colors = ['r','b','g']
        # plotting
        k = 0
        plot_multiple = True if len(directories)>1 else False

        for directory in directories:
            item = dataset_kalman_resampled[directory][index]
            plot = False
            if not files or item[1] in files:
                fig, ax = plt.subplots(figsize=(10, 15))
                data = item[0]
                label = Parsing.parsingLine(sequence=data).getLabelsSequence()
                data_plots.append(plt.plot(data[:, 0]+(k*10), data[:, 1], color=colors[k]))
                ax.scatter(data[:, 0]+(k*10), data[:, 1])
                for i in range(0, len(label)):
                    ax.annotate(str(label[i]), (data[i][0]+(k*10), data[i][1]))
                plot = True
                print(label)
                plt.title(item[1])
                plt.axis('equal')
                plt.show()

        if plot_multiple == True:
            # Legend
            plt.title(item[1])
            plt.axis('equal')
            plt.show()



if debug == 4:
    def test(sequence):
        min_dist = sys.maxsize
        label = None

        for key, values in ideal_sequences.items():
            dist = LevenshteinDistance.levenshtein(ideal_sequences[key], sequence[0])
            if dist < min_dist:
                min_dist = dist
                label = key
        return label
    def get_index(name):
        index = 0
        for label in __classes:
            if label == name:
                return index
            index += 1
        return index

    ideal_sequences = {
        'arrow'                     : ['O', 'A1', 'O', 'A4', 'O', 'A0', 'O', 'A5', 'O'],
        'caret'                     : ['O', 'A1', 'O', 'A7', 'O'],
        'delete_mark'               : ['O', 'A7', 'O', 'A4', 'O', 'A1', 'O'],
        'left_sq_bracket'           : ['O', 'A4', 'O', 'A6', 'O', 'A0', 'O'],
        'rectangle'                 : ['O', 'A6', 'O', 'A0', 'O', 'A2', 'O', 'A4', 'O'],
        'right_sq_bracket'          : ['O', 'A0', 'O', 'A6', 'O', 'A4', 'O'],
        'star'                      : ['O', 'A1', 'O', 'A7', 'O', 'A3', 'O', 'A0', 'O', 'A5', 'O'],
        'triangle'                  : ['O', 'A5', 'O', 'A0', 'O', 'A3', 'O'],
        'v'                         : ['O', 'A7', 'O', 'A1', 'O'],
        'x'                         : ['O', 'A7', 'O', 'A2', 'O', 'A5', 'O']
    }
    directories = ['arrow', 'caret', 'delete_mark', 'left_sq_bracket', 'rectangle', 'right_sq_bracket', 'star', 'triangle', 'v', 'x']

    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/raw/"
    # matrix
    __array = numpy.zeros((len(ideal_sequences), len(ideal_sequences)), dtype=int)
    __classes = list(ideal_sequences.keys())

    for directory in directories:
        # original kalman + resampled
        dataset = CsvDataset(base_dir+directory+"/")
        # kalman
        kalmanTransform = KalmanFilterTransform()
        dataset.addTransform(kalmanTransform)
        # resampling
        #resampledTransform = ResampleTransform
        resampledTransform = ResampleTransform(delta=10)
        dataset.addTransform(resampledTransform)
        # parse
        parse = ParseSamples()
        dataset.addTransform(parse)
        # apply transforms
        for sequence in dataset.applyTransforms():
            print(sequence)
            label = test(sequence)
            __array[get_index(directory)][get_index(label)] += 1

    print(__array)
######################## Debug

