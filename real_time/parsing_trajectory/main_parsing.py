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


debug = 2

if debug == 0:

    random_state = np.random.RandomState(0)

    # Transition matrix
    transition_matrix = [
        [1, 0],
        [0, 1]
    ]
    transition_offset = [0, 0]
    transition_covariance = np.eye(2)
    # Observation matrix
    observation_matrix = [
        [1, 0],
        [0, 1]
    ]
    observation_offset = [0, 0]
    observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
    # Initial state
    initial_state_mean = [0, 0]
    initial_state_covariance = [
        [1, 0],
        [0, 1]
    ]
    # Create Kalman Filter
    kalman_filter = KalmanFilter(
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
    dataset = CsvDataset(dir3)

    # Open files
    for sequence in dataset.readDataset():
        fig, ax = plt.subplots(figsize=(10, 15))
        # Get original sequence 2D
        original_sequence = sequence[0][:,[0,1]]
        # Get smoothed sequence
        smoothed_sequence = kalman_filter.smooth(original_sequence)[0]

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
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/"
    input_dir = base_dir + "raw/"
    output_dir = base_dir + "parsed/"
    directories = ["circle/","rectangle/"]

    for directory in directories:
        dataset = CsvDataset(input_dir+directory)
        # start
        print("Start "+directory)
        sequences = dataset.readDataset()
        len_files = len(sequences)
        for index in range(0,len_files):
            sequence = sequences[index]
            # Get original sequence 2D
            original_sequence = sequence[0][:,[0,1]]
            # Get file name
            name = sequence[1]
            # Parse the sequence and save it
            if not os.path.exists(output_dir+directory):
                os.makedirs(output_dir+directory)
            Parsing.parsingLine(original_sequence, flag_save=True, path=output_dir+directory+name)
        # end
        print("End "+directory)


if debug == 2:
    #parsing = {'A0':'A', 'B0':'U', 'B1':'V', 'B2':'Z', 'B3':'X', 'O0':'O'}
    parsing = {'A0':'A', 'B0':'B', 'B1':'B', 'B2':'B', 'B3':'B', 'O0':'O'}



    def convertList(list_label, dir):
        new_list = []
        for item in list_label:
            string = str(item[0])
            new_list.append(parsing[string])
            # if dir in "circle/":
            #     new_list.append("B")
            # else:
            #     new_list.append("A")
        return new_list

    # % Num train and num test
    quantity_train = 8
    # Get dataset
    base_dir = "/home/ale/PycharmProjects/deictic/repository/deictic/1dollar-dataset/parsed/"
    directories = {
        "circle/": [CsvDataset(base_dir+"circle/")],
        "rectangle/": [CsvDataset(base_dir+"rectangle/")]
    }

    # Create and train hmm
    gesture_hmms = {}
    gesture_datasets = {}
    for directory in directories.keys():
        for dataset in directories[directory]:
            sequences = dataset.readDataset(type=str)
            # create hmm
            model = Model(n_states = 20, n_features = 1, name = directory)
            # get train samples
            num_samples_train = int((len(sequences)*quantity_train)/10)
            samples = [convertList(sequence[0], directory) for sequence in sequences[0:num_samples_train]]
            # train
            model.train(samples)
            # add hmm to dictionary
            gesture_hmms[directory] = [model.getModel()]
            # get test samples
            gesture_datasets[directory] = [convertList(sequence[0], directory) for sequence in sequences[num_samples_train+1:-1]] #[convertList(sequence[0], directory) for sequence in sequences[0:num_samples_train]]
            # debug
            print("Label: " + str(directory) + " - Train :"+str(num_samples_train) + " - Test: "+ str(len(gesture_datasets[directory])))


    #result = Test.getInstance().offlineTest(gesture_hmms=gesture_hmms, gesture_datasets=gesture_datasets, type=str)
    #result.plot()
    # for key, values in gesture_datasets.items():
    #     for value in values:
    #         label, array = Test.compare(value, gesture_hmms, return_log_probabilities=True)
    #         print("Gesture recognized is " + str(label) + " - gesture tested " + key)
    #         print(array)






