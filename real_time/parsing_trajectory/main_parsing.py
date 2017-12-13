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

debug = -1

#dataset = CsvDataset("/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/raw/arrow/")
#dataset2 = CsvDataset("/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/resampled/arrow/")
#dataset.plot(compared_dataset=dataset2, singleMode=True)


if debug == -1:
    # get the gesture expressions which describe 1$ multistroke dataset
    gesture_expressions = DatasetExpressions.returnExpressions(selected_dataset=DatasetExpressions.TypeDataset.unistroke_1dollar)
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/parsed/"
    # get gesture datasets
    gesture_datasets = {
        "arrow": [CsvDataset(base_dir + "arrow/", type=str)],
        "caret": [CsvDataset(base_dir + "caret/", type=str)],
        "check": [CsvDataset(base_dir+"check/", type=str)],
        "circle": [CsvDataset(base_dir+"circle/", type=str)],
        "delete_mark": [CsvDataset(base_dir + "delete_mark/", type=str)],
        "left_curly_brace": [CsvDataset(base_dir + "left_curly_brace/", type=str)],
        "left_sq_bracket": [CsvDataset(base_dir + "left_sq_bracket/", type=str)],
        "pigtail": [CsvDataset(base_dir + "pigtail/", type=str)],
        "question_mark": [CsvDataset(base_dir + "question_mark/", type=str)],
        "rectangle": [CsvDataset(base_dir + "rectangle/", type=str)],
        "right_curly_brace": [CsvDataset(base_dir + "right_curly_brace/", type=str)],
        "right_sq_bracket": [CsvDataset(base_dir + "right_sq_bracket/", type=str)],
         "star": [CsvDataset(base_dir + "star/", type=str)],
        "triangle": [CsvDataset(base_dir + "triangle/", type=str)],
        "v": [CsvDataset(base_dir + "v/", type=str)],
        "x": [CsvDataset(base_dir + "x/", type=str)]
    }
    # hmms
    gesture_hmms = ModelExpression.generatedModels(expressions=gesture_expressions, type=TypeRecognizer.online, num_states=3)

    # for i in range(0, 10):
    #     sample = gesture_hmms["arrow"][0].sample()
    #     print("sample n°"+str(i) + " len:"+str(len(sample)))
    #     print(sample)
    #     print("\n")


    # start log-probability-based test (Test will create the gesture hmms from gesture_expressions)
    results = Test.getInstance().offlineTest(gesture_hmms, gesture_datasets)
    # show result through confusion matrix
    results.plot()

if debug == 1:
    n_sample = 20
    # Get dataset
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/"
    input_dir = base_dir + "raw/"
    output_dir = base_dir + "parsed/"
    directories = [("arrow", 4*n_sample), ("caret", 2*n_sample), ("circle", 4*n_sample), ("check", 2*n_sample), ("delete_mark", 3*n_sample),
                    ("left_curly_brace", 6*n_sample), ("left_sq_bracket", 3*n_sample), ("pigtail", 4*n_sample), ("question_mark", 4*n_sample),
                    ("rectangle", 4*n_sample), ("right_curly_brace", 6*n_sample), ("right_sq_bracket", 3*n_sample), ("star", 5*n_sample),
                    ("triangle", 3*n_sample), ("v", 2*n_sample), ("x", 3*n_sample)]

    for item in directories:
        directory = item[0]
        samples = item[1]

        print("Start " + directory)
        # original kalman + resampled
        dataset_original = CsvDataset(input_dir+directory+"/")
        kalmanTransform = KalmanFilterTransform()
        #resampledTransform = ResampleInSpaceTransform(samples=samples) # accuracy = 0,9448
        resampledTransform = ResampleTransform(delta=6) # accuracy = 0.93 with distance = 5 / accuracy =  with distance = 6 0,9325
        # parse + remove zero
        parse = ParseSamples()
        remove = RemoveZero()
        dataset_original.addTransform(kalmanTransform)
        dataset_original.addTransform(resampledTransform)
        dataset_original.addTransform(parse)
        dataset_original.addTransform(remove)
        sequence_original = dataset_original.applyTransforms(output_dir=(output_dir+directory+"/"))

        print("End " + directory)

        # for index in range(len(sequence_original)):
        #     # Get original sequence 2D
        #     sequence = sequence_original[index][0][:,[0,1]]
        #     # Parse the sequence and save it
        #     if not os.path.exists(output_dir+directory):
        #         os.makedirs(output_dir+directory)
        #     #if "fast" in sequence_original[index][1]:
        #     Parsing.parsingLine(sequence, flag_save=True, path=output_dir+directory+"/"+sequence_original[index][1])



if debug == 2:
    def convertList(list_label):
        new_list = []
        for item in list_label:
            string = str(item[0])
            new_list.append(string)
        return new_list


    # Get dataset
    k_cross_validation = 10
    n_states = 2
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/parsed/"
    directories = {
        "arrow": CsvDataset(base_dir + "arrow/", type=str),
        "caret": CsvDataset(base_dir + "caret/", type=str),
        "check": CsvDataset(base_dir+"check/", type=str),
        "circle": CsvDataset(base_dir+"circle/", type=str),
        "delete_mark": CsvDataset(base_dir + "delete_mark/", type=str),
        "left_curly_brace": CsvDataset(base_dir + "left_curly_brace/", type=str),
        "left_sq_bracket": CsvDataset(base_dir + "left_sq_bracket/", type=str),
        "pigtail": CsvDataset(base_dir + "pigtail/", type=str),
        "question_mark": CsvDataset(base_dir + "question_mark/", type=str),
        "rectangle": CsvDataset(base_dir + "rectangle/", type=str),
        "right_curly_brace": CsvDataset(base_dir + "right_curly_brace/", type=str),
        "right_sq_bracket": CsvDataset(base_dir + "right_sq_bracket/", type=str),
        "star": CsvDataset(base_dir + "star/", type=str),
        "triangle": CsvDataset(base_dir + "triangle/", type=str),
        "v": CsvDataset(base_dir + "v/", type=str),
        "x": CsvDataset(base_dir + "x/", type=str)
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










if debug == 0:
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/raw/"
    directories = ["v"]
    files =[]# ["9_fast_check_09.csv", "5_fast_check_10.csv", "6_fast_check_06.csv", "3_medium_check_01.csv", "11_medium_check_04.csv", "9_medium_check_01.csv", "3_fast_check_07.csv", "3_fast_check_01.csv"]
    dataset_kalman_resampled = {}

    # Raw data with kalman and resampling
    for directory in directories:
        # original kalman + resampled
        dataset_original = CsvDataset(base_dir+directory+"/")
        kalmanTransform = KalmanFilterTransform()
        #resampledTransform = ResampleTransform(delta=4.5)#ResampleInSpaceTransform(samples=40)
        dataset_original.addTransform(kalmanTransform)
        #dataset_original.addTransform(resampledTransform)
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
            if "fast" in item[1]: #item[1] in files or len(files) == 0:
                fig, ax = plt.subplots(figsize=(10, 15))
                data = item[0]
                label = Parsing.parsingLine(sequence=data).getLabelSequences()
                data_plots.append(plt.plot(data[:, 0]+(k*10), data[:, 1], color=colors[k]))
                ax.scatter(data[:, 0]+(k*10), data[:, 1])
                for i in range(0, len(label)):
                    ax.annotate(str(label[i]), (data[i][0]+(k*10), data[i][1]))
                k += 1
                plot = True
                print(label)
            if plot_multiple != True and plot == True:
                # Legend
                plt.title(item[1])
                plt.axis('equal')
                plt.show()
        if plot_multiple == True:
            # Legend
            plt.title(item[1])
            plt.axis('equal')
            plt.show()