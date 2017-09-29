# Libs
import numpy as np
import sys
import types
# Plot
import matplotlib.pyplot as plt

class Result():

    # Methods
    def __init__(self):
        self.data_array = []

    def get(self):
        """

        :param row:
        :return:
        """
        data_to_string = []

        for item in self.data_array:
            if isinstance(item, Result):
                data_to_string.append(item.get())
            else:
                data_to_string.append(str(item))

        return data_to_string

    def getToArray(self):
        """

        :return:
        """
        # Get
        data_to_string = self.get()

        data_to_float = np.zeros(shape=(len(data_to_string),2))
        for row in range(len(data_to_float)):
            for column in range(1):
                data_to_float[row][column] = np.float(data_to_string[row][column])

        return data_to_float


    def save(self, path_to_save):
        """

        :param path_to_save:
        :return:
        """
        data_to_save = []
        for item in self.data_array:
            if isinstance(item, Result):
                data_to_save.append(item.get())

        np.savetxt(path_to_save+'.txt', data_to_save, fmt='%5s', delimiter=',')

class FrameTestResult(Result):
    """
        FrameTestResult describes the results linked to a single frame. It contains the log probabilites computed for each models in the dataset.
    """

    def __init__(self, test_results, n_frame = None):
        super(FrameTestResult, self).__init__()
        ### Check validity parameters ###
        if not isinstance(test_results, list):
            raise ValueError('test_results must be an array.')
        if not isinstance(n_frame, int):
            raise ValueError('n_frame must be an integer.')
        ### Features ###
        # Number of frame, if it is known
        self.n_frame = n_frame
        ## Data structures
        # - log_probabilities contains the log probability value for each model.
        # test_result is an array which its elements have this form: name_model[0] - related log_probability[1]
        #self.log_probabilites = test_results
        self.data_array = test_results
        # - best_log_probability reports which is the model with the highest log probability
        self.best_log_probability = self.__findBestLogProbability()

    def __findBestLogProbability(self):
        best_model = None
        highest_log_probability = -sys.maxsize# (np.finfo(float).eps)

        # Find the highest model
        for item in self.data_array:
            if item[1] > highest_log_probability:
                best_model = item
                highest_log_probability = item[1]

        return best_model

    def get(self):
        """

        :return:
        """
        # Num frame, norm log probability and model name
        return (str(self.n_frame), str(self.best_log_probability[1]), str(self.best_log_probability[0]))


class FileTestResult(Result):
    """
        FileTestResult describes the results linked to a single file.
    """

    def __init__(self, file_path):
        super(FileTestResult, self).__init__()
        ### Check validity parameters ###
        if not isinstance(file_path, str):
            raise ValueError('file_path must be a string.')
        ### Features ###
        # The path of file
        self.file_path = file_path
        ## Data Structures
        # This array contains the test_real_time result of each frame which compose the file
        self.frameTestResults = []
        # This array reports the concise evolution of the recognition
        self.computedSequence = []

    def update(self, data_test):
        """

        :param data_test:
        :return:
        """
        # Create a new FrameTestResult and add it into the correct array
        n_frame = len(self.data_array)+1
        self.data_array.append(FrameTestResult(data_test, n_frame))
        # Update computedSequence array
        self.computeBestSequence()

    def computeBestSequence(self):
        """
            Analyzes the results of all receveid frames.
            In computedSequence, we reported the evolution of the primitives throughout the file,
            highlighting when and how much frames the hmm changes its status.
        :return: None
        """
        # Get the last best model (if computedSequence is not empty) and update the array
        if len(self.computedSequence) > 0:
            last_hmm = self.computedSequence[-1][0]
            if last_hmm != self.data_array[-1].best_log_probability[0]:
                new_hmm = self.data_array[-1].best_log_probability[0]
                start = self.computedSequence[-1][1] + 1
                end = self.computedSequence[-1][2] + 1
                during = 1
                self.computedSequence.append([new_hmm, start, end, during])
            else:
                start = self.computedSequence[-1][1]
                end = self.computedSequence[-1][2] + 1
                during = self.computedSequence[-1][3] + 1
                self.computedSequence[-1] = ([last_hmm, start, end, during])
        else:
            new_hmm = self.data_array[-1].best_log_probability[0]
            start = 0
            end = 1
            during = 1
            self.computedSequence.append([new_hmm, start, end, during])

    def plot(self, filename):
        """

        :return:
        """
        # Get data
        data_to_plot = self.getToArray()
        # Plot
        fig = plt.figure();
        fig, ax = plt.subplots()
        ax.scatter(data_to_plot[:,0], data_to_plot[:,1])
        for i in range(0, len(data_to_plot)):
            ax.annotate(str(i), (data_to_plot[i,0], data_to_plot[i,1]))
        plt.axis('equal')
        plt.plot(data_to_plot[:,0], data_to_plot[:,1])
        plt.title(filename)
        plt.legend(loc='upper right')

        return plt

class DatasetTestResult(Result):
    """
        DatasetTestResult describes the results linked to a whole dataset.
    """

    def __init__(self):
        super(DatasetTestResult, self).__init__()
        ### Features ###


    # Handlers for managing the test_real_time result related to a speficified file's frame.
    def start(self, file_path):
        """
            handler links to the event "starting to 'fire' a new file".
            :return: None
        """
        # Inizialize dictionary
        self.data_array.append(FileTestResult(file_path))

    def update(self, data_test):
        """
            handler links to the event "firing a new frame".
            :return: None
        """
        self.data_array[-1].update(data_test)

    def stop(self):
        """
            handler links to the event "file completed"
            :return: None
        """

    # Override
    def save(self, path_to_save):
        """

        :param path_to_save:
        :return:
        """
        for item in self.data_array:
            item.save(path_to_save+item.file_path)
    # Override
    def plot(self, file_path = None, singleMode = False):
        """

        :return:
        """
        for item in self.data_array:
            plot = item.plot(item.file_path)
            plot.show()

    #
    def check_result(self, n_primitives = 1):
        """

        :return:
        """
        for item in self.data_array:
            # Takes the result about the file
            file_path = item.file_path
            # Number of primitives correctly recognized
            primitive_recognized = 1
            stack = collections.deque()

            # for each row
            for item in file_results:
                # Takes info
                hmm_name = item[0]
                start_frame = int(item[1])
                during_frames = int(item[2])

                # Has it been recognized the correct primitive?
                if str(primitive_recognized) in hmm_name and during_frames > self.__delta_flick:
                    # Update primitive
                    primitive_recognized = self.__updateNumberOfRecognizedPrimitive(primitive_recognized)
                # Update stack
                stack.append(item)

            # Has been recognized the complete gesture?
            if len(stack) == self.num_primitives and primitive_recognized == self.num_primitives:
                self.data_dataset[gesture_name] += 1
            else:
                # print(file_results)
                self.data_dataset_failed[gesture_name].append([filename, file_results])

        # Plot result
        print("Gesture recognized:")