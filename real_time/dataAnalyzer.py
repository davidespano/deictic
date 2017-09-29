# Libs
import sys
import copy
# Fifo
import collections


# Debug lib - csvdataset
from dataset.csvDataset import *
from threading import Thread


class DataAnalyzer():
    """
        This class provides to elaborate and analyze the incoming results about the last examinated file and the dataset which it belongs.
    """
    def __init__(self):
        self.result = []
        # Data for analyzing frame
        self.data_frame = []
        # Data for analyzing a file
        self.data_file = {}
        # Data for analyzing dataset
        self.data_dataset = {}
        self.data_dataset_failed = {}
        # Num Primitive
        self.num_primitives = 0
        # Max flickering permitted
        self.__delta_flick = 0

    def setNumPrimitive(self, num_primitives=0):
        self.num_primitives = num_primitives

    def analyzeFrame(self, frame_log_probabilities):
        """
            Analizes the log probabilities computed for the last managed frame, in order to determine which
            was the hmm with the higher log probability value.
        :param frame_log_probabilities:
        :return: none:
        """
        max_log_probability=-sys.maxsize
        best_hmm = ""
        # Find the hmm with the higher lob probability value
        for item in frame_log_probabilities:
            value = item[1]
            if value > max_log_probability:
                max_log_probability = value
                best_hmm = item[0]
        # Update array data_frame
        self.data_frame.append([best_hmm, max_log_probability])

    def analyzeFile(self, filename):
        """
            Analyzes the results of all frame receveid (which regard a single file).
            In data_file, we reported the evolution of the primitives throughout the file,
            highlighting when and how much frames the hmm changes its status.
        :return: None
        """
        analized_file = []
        start = 0
        during = 0
        hmm_name = self.data_frame[0][0]
        for n_frame in range(0, len(self.data_frame)):
            if hmm_name != self.data_frame[n_frame][0] or n_frame == len(self.data_frame)-1:
                # Saves old pass
                analized_file.append([hmm_name, str(start), str(during)])
                # Updates data
                start = n_frame
                during = 1
                hmm_name = self.data_frame[n_frame][0]
            else:
                during+=1
        # Updates array data_files and clears data_frame
        self.data_file[filename] = copy.deepcopy(analized_file)
        self.data_frame.clear()


    def analyzeOverallData(self, gesture_name):
        """
            Checks whether each sequence of primitives in the dataset has been met.
            Then reports the results in data_dataset.
        :return:
        """

        self.data_dataset_failed[gesture_name] = []
        self.data_dataset[gesture_name] = 0

        # Takes the result about a single file
        for filename in self.data_file:
            file_results = self.data_file[filename]
            # Number of primitives correctly recognized
            primitive_recognized = 1
            stack = collections.deque()

            # for each row
            for item in file_results:
                # Takes info
                hmm_name = item[0]
                start_frame = int(item[1])
                during_frames = int(item[2])

                # Has it been recognized the primitive?
                if str(primitive_recognized) in hmm_name and during_frames > self.__delta_flick:
                    # Update primitive
                    primitive_recognized = self.__updateNumberOfRecognizedPrimitive(primitive_recognized)
                # Update stack
                stack.append(item)

            # Has been recognized the complete gesture?
            if len(stack) == self.num_primitives and primitive_recognized == self.num_primitives:
                self.data_dataset[gesture_name] += 1
            else:
                #print(file_results)
                self.data_dataset_failed[gesture_name].append([filename, file_results])

        # Clears data structures
        self.data_file.clear()


    def __updateNumberOfRecognizedPrimitive(self, num_rec_primitives):
        num_rec_primitives += 1
        if num_rec_primitives > self.num_primitives:
            return self.num_primitives
        else:
            return num_rec_primitives


