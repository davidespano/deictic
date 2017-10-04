# Delay
#from time import sleep
# Analyzing data
from real_time.dataAnalyzer import DataAnalyzer
from real_time.test_real_time import *
# CsvDataset
from dataset import CsvDataset
# Circular Buffer
import collections
# Event
from axel import Event
# Parse
from real_time.modellingGesture import Parse
# Transforms
from dataset.normaliseSamplesDataset import *

# Imports
import numpy as np
import matplotlib.pyplot as plt
import os


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
        self.start = Event() # Start to fire a sequence
        self.fire = Event()  # Fire a frame
        self.end = Event()  # End of sequence
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
        # Takes gesture name from the variable 'dir'
        gesture_name = self.dir.split('/')
        gesture_name = gesture_name[len(gesture_name) - 2]

        for sequence in self.readDataset():
            if filename == None or filename == sequence[1]:
                self.__start(sequence[1])
                for frame in sequence[0]:
                    # Updates buffer and sends frame
                    self.__updateBuffer(frame)
                # Advises that the sequence is completed
                self.__end(sequence[1])


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

    # Firing events methods
    def __start(self, filename):
        self.start(filename)
    def __fire(self, frame):
        self.fire(frame, self.buffer)
    def __end(self, filename):
        # Clear Buffer
        self.__buffer.clear()
        self.end(filename)



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
        # Path save
        self.saveDir = "/home/ale/PycharmProjects/deictic_test/"

        # List of gestures
        if isinstance(type_gesture, str):
            if type_gesture=="unistroke":
                # Gestures
                self.gestures = \
                    [
                        #"unistroke-arrow_1", "unistroke-arrow_2", "unistroke-arrow_3", "unistroke-arrow_4",

                        #"unistroke-circle_1","unistroke-circle_2","unistroke-circle_3","unistroke-circle_4",

                        #"unistroke-check_1","unistroke-check_2",

                        #"unistroke-caret_1","unistroke-caret_2",

                        #"unistroke-delete_mark_1","unistroke-delete_mark_2","unistroke-delete_mark_3",

                        #"unistroke-left_curly_brace_1","unistroke-left_curly_brace_2","unistroke-left_curly_brace_3","unistroke-left_curly_brace_4",

                        #"unistroke-left_sq_bracket_1","unistroke-left_sq_bracket_2","unistroke-left_sq_bracket_3",

                        #"unistroke-pigtail_1","unistroke-pigtail_2","unistroke-pigtail_3","unistroke-pigtail_4",

                        #"unistroke-question_mark_1","unistroke-question_mark_2","unistroke-question_mark_3","unistroke-question_mark_4","unistroke-question_mark_5",

                        #"unistroke-rectangle_1","unistroke-rectangle_2","unistroke-rectangle_3","unistroke-rectangle_4",

                        #"unistroke-right_curly_brace_1","unistroke-right_curly_brace_2","unistroke-right_curly_brace_3","unistroke-right_curly_brace_4",

                        #"unistroke-right_sq_bracket_1","unistroke-right_sq_bracket_2","unistroke-right_sq_bracket_3",

                        #"unistroke-star_1","unistroke-star_2","unistroke-star_3","unistroke-star_4","unistroke-star_5",
                        
                        #"unistroke-triangle_1","unistroke-triangle_2","unistroke-triangle_3",
                        
                        "unistroke-v_1","unistroke-v_2",
                                                
                        #"unistroke-x_1","unistroke-x_2","unistroke-x_3",

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
                        #["arrow", self.n_samples*4],
                        #["circle", self.n_samples*4],
                        #["check", self.n_samples*2],
                        #["caret", self.n_samples * 2],
                        #["delete_mark", self.n_samples * 3],
                        #["left_curly_brace", self.n_samples * 4],
                        #["left_sq_bracket", self.n_samples*3],
                        #["pigtail", n_samples*4],
                        #["question_mark", self.n_samples * 5],
                        #["rectangle", self.n_samples*4],
                        #["right_curly_brace", self.n_samples * 4],
                        #["right_sq_bracket", self.n_samples * 3],
                        #["star", self.n_samples*5],
                        #["triangle", self.n_samples * 3],
                        ["v", self.n_samples*2],
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
        ### Tester ###
        self.test = testRealTime(self.hmms)
        ### Data Collect ###
        self.testResult = {}

        # for each gesture in the dataset
        for gesture in self.gesturesDataset:
            # Gets sequences
            dataset = CsvDatasetRealTime(self.baseDir+gesture[0]+'/', maxlen=dim_buffer, num_samples=gesture[1])
            self.testResult[gesture[0]] = DatasetTestResult(n_primitives = gesture[1]/self.n_samples)

            # Links handlers to events
            dataset.fire += self.test.computeLogProbability # Grabs the fired frame

            dataset.start += self.testResult[gesture[0]].start
            self.test.managing_frame += self.testResult[gesture[0]].update
            #dataset.end += self.testResult[gesture[0]].stop

            # Start firing
            dataset.startFire()
            # Show results
            self.showResults(dataset)

    # Saves results
    def showResults(self, dataset):
        """
            Shows results
        :return:
        """
        for item in self.testResult:
            path = self.saveDir+item+'/'
            if not os.path.exists(path):
                os.makedirs(path)
            self.testResult[item].save(path)
            self.testResult[item].plot(csvDataset=dataset)


#### ####
n_states = 12
n_samples = 40
dim_buffer = 1000
baseDir = '/home/ale/PycharmProjects/deictic/repository/deictic/'
outputDir = '/home/ale/PycharmProjects/deictic/real_time/results/1dollar_dataset/'

app = DeicticRealTime('unistroke', baseDir, outputDir=outputDir, n_states=n_states, n_samples=n_samples, dim_buffer=dim_buffer)
app.application()


#app.printPlot()
#numpy.savetxt(outputDir+'star_result.csv', app.elaborated_results, fmt='%5s', delimiter=',')