#### Library ####
# Csvdataset
#from real_time.parsing_trajectory.csv_dataset import CsvDataset
from dataset.csvDataset import CsvDataset
# Kalman filter
from pykalman import KalmanFilter
# Parsing trajectory definition
from real_time.parsing_trajectory.trajectory_parsing import Parsing
import matplotlib.pyplot as plt
import numpy


debug = 1

if debug == 0:
    # Kalman filther
    # specify parameters
    random_state = numpy.random.RandomState(0)
    transition_matrix = [[1, 0.01], [-0.1, 1]]
    transition_offset = [-0.1, 0.1]
    observation_matrix = numpy.eye(2) + random_state.randn(2, 2) * 0.01
    observation_offset = [1.0, -1.0]
    transition_covariance = numpy.eye(2)
    observation_covariance = numpy.eye(2) + random_state.randn(2, 2) * 0.01
    initial_state_mean = [80, -80]
    initial_state_covariance = [[1, 0.1], [-0.1, 1]]
    # sample from model
    kf = KalmanFilter(
        transition_matrix, observation_matrix, transition_covariance,
        observation_covariance, transition_offset, observation_offset,
        initial_state_mean, initial_state_covariance,
        random_state=random_state
    )

    # Get dataset
    dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/raw/rectangle/"
    dir2 = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/resampled/rectangle/"
    dir3 = "/home/ale/PycharmProjects/deictic/repository/deictic/unica-dataset/raw/right/"
    dir4 = "/home/ale/PycharmProjects/deictic/repository/deictic/unica-dataset/raw/circle/"
    dataset = CsvDataset(dir4)

    # Open files
    for sequence in dataset.readDataset():
        # Get original sequence 2D
        original_sequence = sequence[0][:,[0,1]]
        # Get smoothed sequence
        smoothed_sequence = kf.smooth(original_sequence)[0]

        # Plotting
        # plot original sequence
        original = plt.plot(original_sequence[:,0], original_sequence[:,1], color='b')
        # plot smoothed sequence
        smooth = plt.plot(smoothed_sequence[:, 0], smoothed_sequence[:,1], color='r')
        plt.legend((original[0], smooth[0]), ('true', 'smooth'), loc='lower right')

        plt.show()


if debug == 1:
    # Get dataset
    dir = "/home/ale/PycharmProjects/deictic/repository/deictic/unica-dataset/raw/circle/"
    dataset = CsvDataset(dir)

    for sequence in dataset.readDataset():
        # Get original sequence 2D
        original_sequence = sequence[0][:,[0,1]]

        # Result parsing
        print(sequence[1])
        print(Parsing.getInstance().parsingLine(original_sequence))
        print('\n')





