#### Library ####
# Csvdataset
#from real_time.parsing_trajectory.csv_dataset import CsvDataset
from dataset.csvDataset import CsvDataset
# Kalman filter
from pykalman import KalmanFilter
# Parsing trajectory definition
from real_time.parsing_trajectory.trajectory_parsing import Parsing
import matplotlib.pyplot as plt
import numpy as np


debug = 1

if debug == 0:


    # Get dataset
    dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/raw/rectangle/"
    dir2 = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/resampled/rectangle/"
    dir3 = "/home/ale/PycharmProjects/deictic/repository/deictic/unica-dataset/raw/right/"
    dir4 = "/home/ale/PycharmProjects/deictic/repository/deictic/unica-dataset/raw/circle/"
    dataset = CsvDataset(dir)

    # Open files
    for sequence in dataset.readDataset():
        fig, ax = plt.subplots(figsize=(10, 15))
        # Get original sequence 2D
        original_sequence = sequence[0][:,[0,1]]
        # Get smoothed sequence
        smoothed_sequence = kf.smooth(original_sequence)[0]

        # Plotting
        # plot original sequence
        original = plt.plot(original_sequence[:,0], original_sequence[:,1], color='b')
        # plot smoothed sequence
        smooth = plt.plot(smoothed_sequence[:, 0], smoothed_sequence[:,1], color='r')
        for i in range(0, len(smoothed_sequence)):
            ax.annotate(str(i), (smoothed_sequence[i, 0], smoothed_sequence[i, 1]))
        plt.legend((original[0], smooth[0]), ('true', 'smooth'), loc='lower right')
        plt.title(sequence[1])

        plt.show()


if debug == 1:
    # Get dataset
    dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/resampled/circle/"
    dataset = CsvDataset(dir)

    for sequence in dataset.readDataset():
        # Get original sequence 2D
        original_sequence = sequence[0][:,[0,1]]

        # Result parsing
        Parsing.getInstance().parsingLine(original_sequence)
        print('\n')


