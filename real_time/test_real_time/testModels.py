# Libs
from test.testModels import test
from axel import *

# Copy
import copy

class testRealTime():
    """
        This class implements the methods for testing deictic in real time
    """
    def __init__(self, hmms, plot_result = False):
        # Hmms
        self.hmms = hmms
        # User option regarding the plotting of the computation
        self.plot_result = plot_result
        #### Events ####
        # Is raised when the system completed to fire a file, it is used for sending the results.
        self.managing_frame = Event()

    def computeLogProbability(self, frame, buffer):
        """
            passes the content of buffer to all hmms and returns the norm log-probability of each one.
        :param frame: the frame received
        :param buffer: the list of the latest sent frames
        :return: list of norm log-probabilities
        """
        # Array result
        log_probabilities = []

        for hmm in self.hmms:
            # Computes sequence's log-probability and its normalize
            log_probability = hmm.log_probability(buffer)
            norm_log_probability = log_probability / len(buffer)
            # Update log_probabilities
            log_probabilities.append([hmm.name, norm_log_probability])

            # Print debug results
            #if(self.plot_gesture == True):
            #    print('Model:{} log-probability: {}, normalised-log-probability {}'.
            #          format(hmm.name, log_probability, norm_log_probability))

        # Notifies that the new frame is managed
        self.managing_frame(copy.deepcopy(log_probabilities))

