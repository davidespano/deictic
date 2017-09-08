# Delay
#from time import sleep
# Analyzing data
from real_time.dataAnalyzer import DataAnalyzer
# CsvDataset
from dataset import CsvDataset
# Circular Buffer
import collections
# Event
from axel import Event
# Test
from test import testRealTime
# Parse
from real_time.modellingGesture import Parse
#from gesture.modellingGesture import Parse
# Transforms
from dataset.normaliseSamplesDataset import *

import numpy as np
import matplotlib.pyplot as plt

class CsvDatasetRealTime(CsvDataset):
    """
        Class for firing frame like real time:
        Once it reads the sequence, it fires the frame one by one.
    """

    def __init__(self, dir, maxlen=20, num_samples=20):
        """
        :param dir: dataset's path
        :param maxlen: indicates the dimension of the circular buffer
                       (its size should be big enough to contain all the frame of a gesture)
        """
        # Events
        self.fire = Event()  # Fires a frame
        self.end = Event()  # Ends of sequence
        self.end_dataset = Event() # Ends firing dataset
        # Variables
        self.__buffer = collections.deque(maxlen=maxlen)# Circular buffer
        self.buffer = []#collections.deque(maxlen=maxlen) # Circular buffer
        # Call super method
        super(CsvDatasetRealTime, self).__init__(dir)

        #### Transforms ####
        # Settings items for transforming
        transform1 = NormaliseLengthTransform(axisMode=True)
        transform2 = ScaleDatasetTransform(scale=100)
        transform3 = CenteringTransform()
        #if self.type_gestures == "unistroke":
        transform5 = ResampleInSpaceTransform(samples=num_samples)
        # Apply transforms
        self.addTransform(transform1)
        self.addTransform(transform2)
        self.addTransform(transform3)
        self.addTransform(transform5)

    def applyTransforms(self):
        """
            Applies the transforms only on the saved frames (contain in self.__buffer
        :return: the transformed sequence
        """
        # Passes the buffer as an ndarray
        temp = numpy.asarray(self.__buffer)
        sequence = self.compositeTransform.transform(temp)
        return sequence

    def startFire(self, filename = None):
        """
            For each sequences contained in the dataset
            the system fire each frame of sequence.
        """
        # Sets filename
        gesture_name = self.dir.split('/') # Takes gesture name from the variable 'dir'
        gesture_name = gesture_name[len(gesture_name) - 2]

        for sequence in self.readDataset():
            if filename == None or filename == sequence[1]:
                for frame in sequence[0]:
                    # Updates buffer and sends frame
                    self.__updateBuffer(frame)
                # Advises that the sequence is completed
                self.__end(sequence[1])
        # Advises that the dataset is completed
        self.__endDataset(gesture_name)

    def __updateBuffer(self, frame):
        """
            Provide to update the buffer with the incoming frame
        :return:
        """
        self.__buffer.append(frame)
        if len(self.__buffer) > 5:
            # Applies transforms
            self.buffer = self.applyTransforms()
            # Sends a single frame
            self.__fire(frame)

        #self.buffer.append(frame)# Appends the frame

    # Firing events methods
    def __fire(self, frame):
        self.fire(frame, self.buffer)
    def __end(self, filename):
        # Clear Buffer
        self.__buffer.clear()
        #self.buffer.clear()
        self.end(filename)
    def __endDataset(self, gesture_name):
        self.end_dataset(gesture_name)



class DeicticRealTime():
    """
        Class for firing frame like real time:
        Once it reads the sequence, it fires the frame one by one.
    """

    def __init__(self, type_gesture, baseDir, outputDir = None, n_states = 6, n_samples = 20, dim_buffer = 100):
        # n_states and n_samples
        self.n_states = n_states
        self.n_samples = n_samples
        self.dim_buffer = dim_buffer
        self.outputDir = outputDir
        # Elaborated results
        self.elaborated_results = []

        # List of gestures
        if isinstance(type_gesture, str):
            if type_gesture=="unistroke":
                # Gestures
                self.gestures = \
                    [
                        #"unistroke-v_1",
                        #"unistroke-v_2",
                        #"unistroke-x_1",
                        #"unistroke-x_2",
                        #"unistroke-x_3",
                        "unistroke-rectangle_1",
                        "unistroke-rectangle_2",
                        "unistroke-rectangle_3",
                        "unistroke-rectangle_4",
                        #"unistroke-star_1",
                        #"unistroke-star_2",
                        #"unistroke-star_3",
                        #"unistroke-star_4",
                        #"unistroke-star_5",

                        #"arrow", "caret"
                        #"circle", "check",
                        #"delete_mark", "left_curly_brace",
                        #"left_sq_bracket", "pigtail",
                        #"question_mark", "rectangle",
                        #"right_curly_brace", "right_sq_bracket",
                        #"star", "triangle",
                        #"v", "x"
                    ]
                self.gesturesDataset = \
                    [
                        ["rectangle", self.n_samples*4],
                        #["star", self.n_samples*5],
                        #["v", self.n_samples*2],
                        #["x", self.n_samples*3]

                    ]
                # baseDir (path of dataset folders
                self.baseDir = baseDir+'1dollar-dataset/raw/'
        else:
            # Notifies to user to send a string
            raise ValueError("The gesture type specified it is not valid.")

        # Hmms
        self.hmms = []
        # Creates models
        parse = Parse(n_states, n_samples)
        for gesture in self.gestures:
            # Debug Message
            print(gesture)
            # Gets the hmm
            model = parse.parseExpression(gesture)
            # Sets its name
            model.name = gesture
            # Adds hmm in the list
            self.hmms.append(model)


    def application(self):
        # Inizializing model for analyzing data
        self.data_analyzer = DataAnalyzer()
        # for each gesture in the dataset
        for gesture in self.gesturesDataset:
            # Gets sequences
            dataset = CsvDatasetRealTime(self.baseDir+gesture[0]+'/', maxlen=dim_buffer, num_samples=gesture[1])
            #### Tester ####
            self.test = testRealTime(self.hmms)
            # Links handlers to events
            dataset.fire += self.test.computeLogProbability # Grabs the fired frame
            dataset.end += self.test.compareClassifiers # Computes some information when the file is fired
            #### Data Analyzer ####
            self.test.managing_frame += self.data_analyzer.analyzeFrame # Computes which is the hmm with the higher log probability value
            dataset.end += self.data_analyzer.analyzeFile # Computes the evolution of the primitives throughout the file.
            dataset.end_dataset += self.data_analyzer.analyzeOverallData # Computes how many file are recognized correctly.
            self.data_analyzer.setNumPrimitive(len(self.hmms))

            #self.test.updateResults += self.__savesResults
            # Start firing
            dataset.startFire()

    # Saves results
    def showResults(self):
        """

        :return:
        """
        # Shows the analyzed results
        for gesture in self.data_analyzer.data_dataset:
            # Takes data
            data = self.data_analyzer.data_dataset[gesture]
            # Prints gesture name, positive true, negative true
            print(gesture +' - '+ str(data) +'/'+ (str(330-data)))

    # Plots results
    # Handler for test model event
    def __plotResults(self, log_probabilities, results_for_file, file_name):
        self.results.append(log_probabilities)
    # Plots the results
    def printPlot(self):
        for result in self.results:
            plt.clf()
            # Plot
            for hmm in self.hmms:
                y = result[hmm.name]
                x = np.arange(len(y))
                plt.plot(x,y, label=hmm.name)

            plt.legend(bbox_to_anchor=(.05, 1), loc='best', borderaxespad=0.)
            plt.show()

#### ####
n_states = 6
n_samples = 20
dim_buffer = 1000
baseDir = '/home/ale/PycharmProjects/deictic/repository/deictic/'
outputDir = '/home/ale/PycharmProjects/deictic/real_time/results/1dollar_dataset/'

app = DeicticRealTime('unistroke', baseDir, outputDir=outputDir, n_states=n_states, n_samples=n_samples, dim_buffer=dim_buffer)
app.application()
app.showResults()


#app.printPlot()
#numpy.savetxt(outputDir+'star_result.csv', app.elaborated_results, fmt='%5s', delimiter=',')
