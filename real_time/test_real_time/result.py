# Libs
import numpy as np
import sys
import types
# Plot
import matplotlib.pyplot as plt

class Result():

    # Methods
    def __init__(self, n_primitives=1):
        self.data_array = []
        # Features
        self.n_primitives = n_primitives

    def get(self):
        """
            converts data_array to string.
        :return:
        """
        data_to_string = []

        for item in self.data_array:
            if isinstance(item, Result):
                data_to_string.append(item.get())
            else:
                data_to_string.append(str(item))

        return data_to_string

    def save(self, path_to_save):
        """
            writes the collected data in the specified file.
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

    def __init__(self, test_results, n_frame = None, n_primitives = 1):
        super(FrameTestResult, self).__init__(n_primitives)
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
        """
            find which is the best model in the last result.
        :return: the model with the highest norm log probability value.
        """
        best_model = None
        highest_log_probability = -sys.maxsize# (np.finfo(float).eps)

        # Find the highest model
        for item in self.data_array:
            if item[1] > highest_log_probability:
                best_model = item
                highest_log_probability = item[1]

        return best_model

    # Override
    def get(self):
        """
            converts data_array to string.
        :return:
        """
        # Num frame, norm log probability and model name
        return (str(self.n_frame), str(self.best_log_probability[1]), str(self.best_log_probability[0]))

    def resultsToArray(self):
        """
            converts data_array to string.
        :return:
        """
        data = {}
        for item in self.data_array:
            data[item[0]] = [item[1]]
        return data

class FileTestResult(Result):
    """
        FileTestResult describes the results linked to a single file.
    """

    def __init__(self, file_path, n_primitives = 1):
        super(FileTestResult, self).__init__(n_primitives)
        ### Check validity parameters ###
        if not isinstance(file_path, str):
            raise ValueError('file_path must be a string.')
        ### Features ###
        # The path of file
        self.file_path = file_path
        ## Data Structures
        # This array reports the concise evolution of the recognition
        self.computedSequence = []
        # This array reports for each models the log probabilitie values during the evolution
        self.testTrend = {}

    def update(self, data_test):
        """
            creates a new FrameTestResult and passes to it the incoming results.
        :param data_test:
        :return:
        """
        # Create a new FrameTestResult and add it into the correct array
        n_frame = len(self.data_array)+1
        self.data_array.append(FrameTestResult(data_test, n_frame))
        # Update computedSequence array
        self.computeBestSequence()
        # Update test trend
        self.collectModelsTrend()

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

    def collectModelsTrend(self):
        """

        :return:
        """
        for key,value in self.data_array[-1].resultsToArray().items():
            if key not in self.testTrend:
                self.testTrend[key] = []
            self.testTrend[key].append(value)

    def check(self):
        """
            This method analyzes the data include into computedSequence in order to check if primitives have been recognized correctly.
        :return:
            yes (the primitives have been recognized correctly) or false.
        """
        primitive_recognized = 1
        for item in self.computedSequence:
            # Gets info
            model_name = item[0]
            start_frame = item[1]
            end_frame = item[2]
            during_frames = item[3]

            # Has it been recognized the correct primitive?
            if str(primitive_recognized) in hmm_name and during_frames > self.__delta_flick:
                # Update primitive
                primitive_recognized = self.__updateNumberOfRecognizedPrimitive(primitive_recognized)

        # Has been recognized the complete gesture? If yes return true else false
        if len(stack) == self.n_primitives and primitive_recognized == self.n_primitives:
            return True
        else:
            return False

    def plot(self):
        """
            this method collects the log probabilities of each model and creates a plot of these data.
        :return:
        """
        plot = plt.figure(1, figsize=(10,20))
        # Plot
        for key,value in self.testTrend.items():
            x = np.arange(len(self.data_array))
            y = value
            plt.plot(x, y, label=key)

        # Title
        plt.title(self.file_path)
        # Stretching x
        plt.axes([min(x), max(x), -110, 0])
        # Ticks on x axis
        #plt.xticks(np.arange(min(x), max(x) + 1, 2.0))
        # Legend
        plt.legend(bbox_to_anchor=(.05, 1), loc='best', borderaxespad=0.)
        return plot



class DatasetTestResult(Result):
    """
        DatasetTestResult describes the results linked to a whole dataset.
    """

    def __init__(self, n_primitives=1):
        super(DatasetTestResult, self).__init__(n_primitives)
        #### Features ####
        # For each file, test_result reports if the primitives have been recognized correctly or not.
        self.test_result = {}

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

    def stop(self, file_path):
        """
            handler links to the event "file completed"
            :return: None
        """
        self.test_result[file_path] = self.data_array[-1].check()

    # Override
    def save(self, path_to_save):
        """
            saves the data in the specified file.
        :param path_to_save:
        :return:
        """
        for item in self.data_array:
            item.save(path_to_save+item.file_path)
    # Override
    def plot(self, csvDataset = None):
        """
            plots the analized data.
        :return:
        """
        for item in self.data_array:
            plot = item.plot()
            plot.show()
            # If csvDataset is not None, plots also the file
            csvDataset.plot(sampleName=item.file_path)