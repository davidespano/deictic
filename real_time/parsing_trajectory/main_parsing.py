#### Library ####
# Csvdataset
#from real_time.parsing_trajectory.csv_dataset import CsvDataset
from dataset.csvDataset import CsvDataset
# Test
from test.test import Test, Result
# Kalman filter
from pykalman import KalmanFilter
# Parsing trajectory definition
from real_time.parsing_trajectory.trajectory_parsing import Parsing
from real_time.parsing_trajectory.model_factory import Model
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import *
import random
import math
import datetime
import time

debug = 2

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
        resampledTransform = ResampleTransform(delta=6)#ResampleInSpaceTransform(samples=40)
        dataset_original.addTransform(kalmanTransform)
        dataset_original.addTransform(resampledTransform)
        dataset_kalman_resampled[directory] = dataset_original.applyTransforms()

    for index in range(len(dataset_kalman_resampled[directory])):
        data_plots = []
        colors = ['r','b','g']
        # plotting
        k = 0
        plot = False
        fig, ax = plt.subplots(figsize=(10, 15))
        for directory in directories:
            item = dataset_kalman_resampled[directory][index]
            if item[1] in files or len(files) == 0:
                data = item[0]
                label = Parsing.parsingLine(sequence=data).getLabelSequences()
                data_plots.append(plt.plot(data[:, 0]+(k*10), data[:, 1], color=colors[k]))
                ax.scatter(data[:, 0]+(k*10), data[:, 1])
                for i in range(0, len(label)):
                    ax.annotate(str(label[i]), (data[i][0]+(k*10), data[i][1]))
                k += 1
                plot = True
        if plot == True:
            # Legend
            plt.legend((data_plots), (directories), loc='lower right')
            plt.axis('equal')
            plt.show()


if debug == 1:
    # Get dataset
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/"
    input_dir = base_dir + "raw/"
    #input_dir = base_dir + "resampled/"
    output_dir = base_dir + "parsed/"
    directories = ["check", "v"]
    #directories = ["arrow", "caret", "check", "circle", "delete_mark", "left_curly_brace", "left_sq_bracket", "pigtail", "question_mark", "rectangle",
    #               "right_curly_brace", "right_sq_bracket", "star", "triangle", "v", "x"]

    for directory in directories:
        # original kalman + resampled
        dataset_original = CsvDataset(input_dir+directory+"/")
        kalmanTransform = KalmanFilterTransform()
        resampledTransform = ResampleTransform(delta=6) #ResampleInSpaceTransform(samples=40)
        dataset_original.addTransform(kalmanTransform)
        dataset_original.addTransform(resampledTransform)
        sequence_original = dataset_original.applyTransforms()

        # start
        print("Start "+directory)
        for index in range(len(sequence_original)):
            # Get original sequence 2D
            sequence = sequence_original[index][0][:,[0,1]]
            # Parse the sequence and save it
            if not os.path.exists(output_dir+directory):
                os.makedirs(output_dir+directory)
            Parsing.parsingLine(sequence, flag_save=True, path=output_dir+directory+"/"+sequence_original[index][1])
        # end
        print("End "+directory)


if debug == 2:
    def convertList(list_label):
        new_list = []
        for item in list_label:
            string = str(item[0])
            new_list.append(string)
        return new_list


    # Get dataset
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/parsed/"
    directories = {
        # "arrow": CsvDataset(base_dir + "arrow/", type=str),
        #"caret": CsvDataset(base_dir + "caret/", type=str),
        "check": CsvDataset(base_dir+"check/", type=str),
        # "circle": CsvDataset(base_dir+"circle/", type=str),
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

    # Create and train hmm
    start_time = time.time()#### debug time
    gesture_hmms = {}
    gesture_datasets = {}
    for directory in directories.keys():
            # create hmm
            model = Model(n_states = 15, n_features = 1, name = directory)
            # get train and test samples
            train_samples,test_samples = directories[directory].crossValidation()
            model.train(train_samples)
            gesture_datasets[directory] = test_samples
            # add hmm to dictionary
            gesture_hmms[directory] = [model.getModel()]
            # debug
            print("Label: " + str(directory) + " - Train :"+str(len(train_samples)) + " - Test: "+ str(len(test_samples)))


    result = Test.getInstance().offlineTest(gesture_hmms=gesture_hmms, gesture_datasets=gesture_datasets)
    print("--- %s seconds ---" % (time.time() - start_time)) # debug time
    result.plot()
    print(result.getWrongClassified())
    # for key, values in gesture_datasets.items():
    #     for value in values:
    #         label, array = Test.compare(value, gesture_hmms, return_log_probabilities=True)
    #         print("Gesture recognized is " + str(label) + " - gesture tested " + key)
    #         print(array)
