import numpy
import sys
import collections
# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import itertools
# csv
import csv
# Deictic
from gesture import ModelExpression
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
        self.__labels = sorted(gesture_labels)
        # array (the core of the confusion matrix)
        self.__array = numpy.zeros((len(self.__labels), len(self.__labels)), dtype=int)
        # array (if the sequences have a name it contains, for each model, the list of wrong recognized files)
        self.__dictionary = {}
        for item in self.__labels:
            self.__dictionary[item] = []

    def getWrongClassified(self):
        return self.__dictionary

    def update(self, row_label, column_label, id_sequence = None):
        """
            this method autoincrements the specified index of the matrix confusion and, eventually, updates the dictionary.
        :param row: the dataset from which the file came (str).
        :param coloumn: the gesture which recognized this file (str).
        :return:
        """
        if not isinstance(row_label, str) or not isinstance(column_label, str):
            raise Exception("The parameters must be string.")
        row = self.__getLabelIndex(wanted_label=row_label)
        column = self.__getLabelIndex(wanted_label=column_label)
        self.__array[row][column]+=1
        # update dictionary of wrong recognized files
        if id_sequence != None and row_label != column_label:
            self.__dictionary[row_label].append([id_sequence,column_label])

    def meanAccuracy(self):
        """
            this methods computes and returns the mean accuracy obtained from the test.
        :return:
        """
        mean_accuracy = 0
        # The mean accurancy is computed by using the data returned from self.detailedResults
        means = self.__getAccuracyModels()
        for model_name in self.__labels:
            mean_accuracy+=means[model_name][-1]
        return mean_accuracy/len(self.__labels)


    def save(self, path):
        """
            this methods saves the array contents in the specified path.
        :param path:
        :return:
        """
        if not isinstance(path, str):
            raise Exception("The parameter path must be string.")
        array_to_save = numpy.empty([len(self.__labels)+1, len(self.__labels)+1], dtype=object)
        # Gesture labels
        for index in range(0, len(self.__labels)):
            array_to_save[0][index+1] = self.__labels[index]
            array_to_save[index+1][0] = self.__labels[index]
        # Array values
        for row in range(0,len(self.__labels)):
            for column in range(0,len(self.__labels)):
                array_to_save[row+1][column+1] = self.__array[row,column]
        # Save confusion matrix
        #array_to_save.tofile(path)
        # Save confusion matrix
        with open(path, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in array_to_save:
                spamwriter.writerow(row)

    def plot(self, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
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

        plt.rcParams.update({'font.size': 8})
        plt.imshow(self.__array, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = numpy.arange(len(self.__labels))
        plt.xticks(tick_marks, self.__labels, rotation=45)
        plt.yticks(tick_marks, self.__labels)

        if normalize:
            den = self.__array.sum(axis=1)[:, numpy.newaxis]
            print(den)
            self.__array = self.__array.astype('float') / den[0]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(self.__array)

        thresh = self.__array.max() / 2.
        for i, j in itertools.product(range(self.__array.shape[0]), range(self.__array.shape[1])):
            if (self.__array[i, j] >= 0.01):
                plt.text(j, i, self.__array[i, j],
                         horizontalalignment="center",
                         color="white" if self.__array[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Mean Accuracy '+str(self.meanAccuracy()))
        print('Mean Accuracy: '+str(self.meanAccuracy()))
        plt.show()

    def __add__(self, other):
        """
            The sum operator allows the developers to add the results of differents iterations (of course, the two objects have to go the same labels).
        :param other:
        :return:
        """
        # Check parameters
        if not isinstance(other, Result):
            raise TypeError
        compare = lambda x, y: collections.Counter(x) == collections.Counter(y) # this function is used for comparing two list of objects
        if not compare(other.__labels, self.__labels):
            raise Exception
        # add arrays
        self.__array = numpy.sum([self.__array, other.__array], axis=0)
        return self


    ### Private methods ###
    def __getLabelIndex(self, wanted_label):
        """
            this methods finds and returns the index links to the passed label.
        :param label: a gesture label
        :return: the corresponding index or raise an exception if labels does not contain wanted_label.
        """
        index = 0
        for label in self.__labels:
            if label == wanted_label:
                return index
            index+=1
        raise Exception(wanted_label+" is not present.")
    def __getAccuracyModels(self):
        """
            this methods computes and returns a more detailed analysis about the results obtained for each model.
        :return: a triple of [the number of elements recognized correctly, the number of elements used for the test and mean and the accuracy]
        """
        means = {}
        # for each models
        for model_name in self.__labels:
            # get row index
            row = self.__getLabelIndex(model_name)
            # compute true and false results
            total = self.__array[row,:].sum()
            true = self.__array[row][row]
            # save mean
            means[model_name] = [true, total, true/total]
        return means

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

    def offlineTestExpression(self, gesture_expressions, gesture_datasets):
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

    def offlineTest(self, gesture_hmms, gesture_datasets):
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
                    sequences = [sequence for sequence in dataset.readDataset()]
                elif isinstance(dataset, (numpy.ndarray, list)):
                    sequences = [dataset]
                else:
                    raise Exception("gesture dataset must be a CsvDataset object or a numpy ndarray!")
                # Comparison
                self.__comparison(sequences=sequences, dataset_label=gesture_label)
        # return comparison results
        return self.result

    def onlineTestExpression(self, gesture_expressions, gesture_datasets):
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
        return ModelExpression.generatedModels(expressions=gesture_expressions)

    def __comparison(self, sequences, dataset_label):
        """
            given a list of sequence, this method updates the result array based on the comparison of each model.
        :param sequences: a list of sequence frames.
        :param dataset_label: the list of gesture labels.
        :return:
        """
        # Get each sequence
        for sequence in sequences:
            index_label = Test.compare(sequence[0], self.gesture_hmms)
            # Update results
            if index_label != None:
                self.result.update(row_label=dataset_label, column_label=index_label, id_sequence=sequence[1])

    @staticmethod
    def compare(sequence, gesture_hmms, return_log_probabilities = False):
        """
            given a sequence, this method computes the log probability for each model and returns the model with the highest norm log probability.
        :param sequence: a sequence of frame which describes an user movement.
        :param gesture_hmms: a dictionary of hidden markov models ({'gesture_label':[hmm_1, hmm_2, ....]).
        :param return_log_probabilities: this boolean indicates the label of the best compared model or also the norm log probabilities of each model for the passed sequence.
        :return: the model label with the best norm log probability,
        """
        # Max probability "global"
        max_norm_log_probability = -sys.maxsize
        # Model label with the best norm log proability for that sequence
        index_label = None
        # Log probability values
        log_probabilities = {}

        #print("\n")
        # Compute log probability for each model
        for gesture_label, models in gesture_hmms.items():
            # Max probability "local"
            local_norm_log_probabilty = -sys.maxsize
            for model in models:
                # Compute sequence's log-probability and normalized
                log_probability = model.log_probability(sequence)
                norm_log_probability = log_probability / len(sequence)
                #print(gesture_label+": "+str(norm_log_probability))
                # Check which is the best 'local' model
                if norm_log_probability > local_norm_log_probabilty:
                    local_norm_log_probabilty = norm_log_probability
            # Collect additional data, if necessary (whether additional_data is true)
            if return_log_probabilities:
                log_probabilities[gesture_label] = local_norm_log_probabilty
            # Check which is the best model
            if (local_norm_log_probabilty > max_norm_log_probability):
                max_norm_log_probability = norm_log_probability
                index_label = gesture_label
        # Comparison completed, index_label contains the best global model while log_probabilities the norm log probabilities of each model for the passed sequence.
        #print(index_label)
        if not return_log_probabilities:
            return index_label
        else:
            return index_label, log_probabilities