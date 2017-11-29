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

import math

import time

debug = 2

if debug == 1:
    # Get dataset
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/"
    input_dir = base_dir + "raw/"
    #input_dir = base_dir + "resampled/"
    output_dir = base_dir + "parsed/"
    #directories = ["triangle"]
    directories = ["arrow", "caret", "check", "circle", "delete_mark", "left_curly_brace", "left_sq_bracket", "pigtail", "question_mark", "rectangle",
                   "right_curly_brace", "right_sq_bracket", "star", "triangle", "v", "x"]

    for directory in directories:
        # original kalman + resampled
        dataset_original = CsvDataset(input_dir+directory+"/")
        kalmanTransform = KalmanFilterTransform()
        resampledTransform = ResampleInSpaceTransform(samples=40)
        dataset_original.addTransform(kalmanTransform)
        dataset_original.addTransform(resampledTransform)
        sequence_original = dataset_original.applyTransforms()

        # start
        print("Start "+directory)
        for index in range(len(sequence_original)):
            # Get original sequence 2D
            sequence = sequence_original[index][:,[0,1]]
            # Get file name
            #name = sequence_original[index][1]
            # Parse the sequence and save it
            if not os.path.exists(output_dir+directory):
                os.makedirs(output_dir+directory)
            Parsing.parsingLine(sequence, flag_save=True, path=output_dir+directory+"/"+str(index)+".csv")
        # end
        print("End "+directory)


if debug == 2:
    def convertList(list_label):
        new_list = []
        for item in list_label:
            string = str(item[0])
            new_list.append(string)
        return new_list


    # % Num train and num test
    quantity_train = 9
    # Get dataset
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/parsed/"
    directories = {
        "arrow": CsvDataset(base_dir + "arrow/"),
        "caret": CsvDataset(base_dir + "caret/"),
        "check": CsvDataset(base_dir+"check/"),
        "circle": CsvDataset(base_dir+"circle/"),
        "delete_mark": CsvDataset(base_dir + "delete_mark/"),
        "left_curly_brace": CsvDataset(base_dir + "left_curly_brace/"),
        "left_sq_bracket": CsvDataset(base_dir + "left_sq_bracket/"),
        "pigtail": CsvDataset(base_dir + "pigtail/"),
        "question_mark": CsvDataset(base_dir + "question_mark/"),
        "rectangle": CsvDataset(base_dir + "rectangle/"),
        "right_curly_brace": CsvDataset(base_dir + "right_curly_brace/"),
        "right_sq_bracket": CsvDataset(base_dir + "right_sq_bracket/"),
        "star": CsvDataset(base_dir + "star/"),
        "triangle": CsvDataset(base_dir + "triangle/"),
        "v": CsvDataset(base_dir + "v/"),
        "x": CsvDataset(base_dir + "x/")
    }

    # Create and train hmm
    start_time = time.time()#### debug time
    gesture_hmms = {}
    gesture_datasets = {}
    for directory in directories.keys():
            sequences = directories[directory].readDataset(type=str)
            # create hmm
            model = Model(n_states = 15, n_features = 1, name = directory)
            # get train samples
            num_samples_train = int((len(sequences)*quantity_train)/10)
            samples = [convertList(sequence[0]) for sequence in sequences[0:num_samples_train]]
            # train
            model.train(samples)
            # add hmm to dictionary
            gesture_hmms[directory] = [model.getModel()]
            # get test samples
            gesture_datasets[directory] = [convertList(sequence[0]) for sequence in sequences[num_samples_train+1:-1]] #[convertList(sequence[0], directory) for sequence in sequences[0:num_samples_train]]
            # debug
            print("Label: " + str(directory) + " - Train :"+str(num_samples_train) + " - Test: "+ str(len(gesture_datasets[directory])))


    result = Test.getInstance().offlineTest(gesture_hmms=gesture_hmms, gesture_datasets=gesture_datasets, type=str)
    print("--- %s seconds ---" % (time.time() - start_time)) # debug time
    result.plot()
    # for key, values in gesture_datasets.items():
    #     for value in values:
    #         label, array = Test.compare(value, gesture_hmms, return_log_probabilities=True)
    #         print("Gesture recognized is " + str(label) + " - gesture tested " + key)
    #         print(array)