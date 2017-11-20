import numpy
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import itertools
from gesture import ModelFactory
from dataset import CsvDataset

class Result():
    """
        Results provides the methods for managing, plotting and saving confusion matrix.
    """

    ### Public methods ###
    def __init__(self, gesture_labels = []):
        # Check parameters
        if not isinstance(gesture_labels, list):
            raise Exception("name_gestures must be a list of object.")
        # gesture_labels
        self.labels = gesture_labels
        # array (the core of the confusion matrix)
        self.array = numpy.zeros((len(self.labels), len(self.labels)), dtype=int)

    def update(self, row_label, column_label):
        """
            this method autoincrements the specified index.
        :param row: the dataset from which the file came (str).
        :param coloumn: the gesture which recognized this file (str).
        :return:
        """
        if not isinstance(row_label, str) or not isinstance(column_label, str):
            raise Exception("The parameters must be string.")
        row = self.__getLabelIndex(wanted_label=row_label)
        column = self.__getLabelIndex(wanted_label=column_label)
        self.array[row][column]+=1

    def save(self, path):
        """
            this methods saves the array contents in the specified path.
        :param path:
        :return:
        """
        if not isinstance(path, str):
            raise Exception("The parameter path must be string.")
        array_to_save = numpy.chararray((len(self.labels)+1, len(self.labels)+1))
        # Gesture labels
        for index in range(0, len(self.labels)):
            array_to_save[index+1,index+1] = self.labels[index]
            index+1
        # Array values
        for row in range(0,len(self.labels)):
            for column in range(0,len(self.labels)):
                array_to_save[row+1, column+1] = self.array[row,column]
        # Save confusion matrix
        array_to_save.tofile(path)

    def plot(self, normalize=False, title="Confusion Matrix", cmap=plt.cm.Greens):
        """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
        :param normalize:
        :param title:
        :param cmap:
        :return:
        """
        if not isinstance(normalize, bool):
            raise Exception("normalize must be bool.")
        if not isinstance(title, str):
            raise Exception("title must be bool.")
        if not isinstance(cmap, LinearSegmentedColormap):
            raise Exception("cmap must be a LinearSegmentedColormap object.")

        plt.imshow(self.array, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = numpy.arange(len(self.labels))
        plt.xticks(tick_marks, self.labels, rotation=45)
        plt.yticks(tick_marks, self.labels)

        if normalize:
            den = self.array.sum(axis=1)[:, numpy.newaxis]
            print(den)
            self.array = self.array.astype('float') / den[0]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(self.array)

        thresh = self.array.max() / 2.
        for i, j in itertools.product(range(self.array.shape[0]), range(self.array.shape[1])):
            if (self.array[i, j] >= 0.01):
                plt.text(j, i, self.array[i, j],
                         horizontalalignment="center",
                         color="white" if self.array[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    ### Private methods ###
    def __getLabelIndex(self, wanted_label):
        """
            this methods finds and returns the index links to the passed label.
        :param label: a gesture label
        :return: the corresponding index or raise an exception if labels does not contain wanted_label.
        """
        index = 0
        for label in self.labels:
            if label == wanted_label:
                return index
            index+=1
        raise Exception(wanted_label+" is not present.")

class Test():

    # singleton
    __singleton = None

    ### Public methods ###
    @staticmethod
    def getInstance():
        """
        :return:
        """
        if Test.__singleton == None:
            Test.__singleton = Test()
        return Test.__singleton

    def offlineTestExpression(self, gesture_expressions, gesture_datasets, type=float):
        """
            offlineTestExpression creates the models, starting from the passed expressions, and starts the comparison.
        :param gesture_expressions: a dictionary of deictic expressions (key is the gesture label, values are his expressions).
        :param gesture_datasets: is a dictionary of CsvDataset objects (key is the gesture label, value is the linked dataset).
        :return: comparison results
        """
        # Check parameters
        if not isinstance(gesture_expressions, dict):
            raise Exception("gesture_expressions must be a dictionary of deictic expressions.")
        # creates models
        gesture_hmms = self.__createModel(gesture_expressions=gesture_expressions)
        # start comparison
        return self.offlineTest(gesture_hmms=gesture_hmms, gesture_datasets=gesture_datasets, type=float)

    def offlineTest(self, gesture_hmms, gesture_datasets, type=float):
        """
            offlineTest compares gesture hmms each one and return the resulted confusion matrix.
        :param gesture_hmms: a dictionary of hidden markov models (key is the gesture label, values are his expression models).
        :param gesture_datasets: is a dictionary of CsvDataset objects (key is the gesture label, value is the linked dataset).
        :return: comparision results
        """
        # Check parameters

        if not isinstance(gesture_hmms, dict):
            raise Exception("gesture_hmms must be a dictionary of hidden markov models.")
        if not isinstance(gesture_datasets, dict):
            raise Exception("gesture_datasets must be a dictionary of CsvDataset objects.")
        if len(gesture_hmms) != len(gesture_datasets):
            raise Exception("gesture_hmms and gesture_datasets must have the same lenght.")

        self.gesture_hmms = gesture_hmms
        self.gesture_datasets = gesture_datasets
        # comparison results
        self.result = Result(list(self.gesture_hmms.keys()))

        # comparasing gesture_hmms
        for gesture_label, datasets in self.gesture_datasets.items():
            # for each dataset
            for dataset in datasets:
                # csvDataset or list of sequences?
                if isinstance(dataset, CsvDataset):
                    sequences = [sequence[0] for sequence in dataset.readDataset(type=type)]
                elif isinstance(dataset, numpy.ndarray):
                    sequences = [dataset]
                else:
                    raise Exception("gesture dataset must be a CsvDataset object or a numpy ndarray!")
                # Comparison
                self.__comparison(sequences=sequences, dataset_label=gesture_label)
        # return comparison results
        return self.result

    def onlineTestExpression(self, gesture_expressions, gesture_datasets, type=float):
        """
            onlineTestExpression creates the models, starting from the passed expressions, and starts the comparison.
        :param gesture_expressions: a dictionary of deictic expressions (key is the gesture label, values are his expressions).
        :param gesture_datasets: is a dictionary of CsvDataset objects (key is the gesture label, value is the linked dataset).
        :return: comparison results
        """
        # creates models
        gesture_hmms = self.__createModel(gesture_expressions=gesture_expressions)
        # start comparison
        return self.onlineTest(gesture_hmms=gesture_hmms, gesture_datasets=gesture_datasets, type=float)

    def onlineTest(self, gesture_hmms, gesture_datasets, type=float): pass

    ### Private methods ###
    def __createModel(self, gesture_expressions):
        """
            this method creates a hmm for each expression in gesture_expressions.
        :param gesture_expressions: a dictionary of deictic expression.
        :return: a dictionary of deictic hmms.
        """
        return ModelFactory.createHmm(expressions=gesture_expressions)

    def __comparison(self, sequences, dataset_label):
        """
            given a list of sequence, this method updates the result array based on the comparison of each model.
        :param sequences: a list of sequence frames.
        :param dataset_label: the list of gesture labels.
        :return:
        """
        # Get each sequence
        for sequence in sequences:
            index_label = Test.__compare(sequence, self.gesture_hmms)
            # Update results
            if index_label != None:
                self.result.update(row_label=dataset_label, column_label=index_label)

    @staticmethod
    def __compare(sequence, gesture_hmms):
        """
            given a sequence, this method computes the log probability for each model and returns the model with the highest norm log probability.
        :param sequence:
        :param gesture_hmms:
        :return:
        """
        # Max probability
        max_norm_log_probability = -sys.maxsize
        index_label = None
        # Compute log probability for each model
        for gesture_label, models in gesture_hmms.items():
            for model in models:
                # Computes sequence's log-probability and normalized
                log_probability = model.log_probability(sequence)
                norm_log_probability = log_probability / len(sequence)
                # Checks which is the best result
                if (norm_log_probability > max_norm_log_probability):
                    max_norm_log_probability = norm_log_probability
                    index_label = gesture_label
        return index_label
