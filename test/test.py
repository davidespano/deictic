import numpy
import csv
import matplotlib as plt
import itertools

class Result():
    """
        Results provides the methods for managing, plotting and saving confusion matrix.
    """

    ### Public methods ###
    def __init__(self, gesture_labels = []):
        # Check parameters
        if not isinstance(gesture_labels, list):
            raise ("name_gestures must be a list of object.")
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
        row = self.__getLabelIndex(wanted_label=row_label)
        column = self.__getLabelIndex(wanted_label=column_label)
        self.array[row, column]+=1

    def save(self, path):
        """
            this methods saves the array contents in the specified path.
        :param path:
        :return:
        """
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
        plt.imshow(self.array, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = numpy.arange(len(self.labels))
        plt.xticks(tick_marks, self.labels, rotation=45)
        plt.yticks(tick_marks, self.labels)

        if normalize:
            den = self.array.sum(axis=1)[:, numpy.newaxis]
            print(den)
            cm = self.array.astype('float') / den[0]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if (cm[i, j] >= 0.01):
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    ### Private methods ###
    def __getLabelIndex(self, wanted_label):
        """
            this methods finds and returns the index links to the passed label.
        :param label: a gesture label
        :return:
        """
        index = 0
        for label in self.labels:
            if label == wanted_label:
                return index
            index+=1

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
        # creates models
        self.__createModel(gesture_expressions=gesture_expressions)
        # start comparison
        self.offlineTest(gesture_hmms=gesture_hmms, gesture_datasets=gesture_datasets)

    def offlineTest(self, gesture_hmms, gesture_datasets):
        """
            offlineTest compares gesture hmms each one and return the resulted confusion matrix.
        :param gesture_hmms: a dictionary of hidden markov models (key is the gesture label, values are his expression models).
        :param gesture_datasets: is a dictionary of CsvDataset objects (key is the gesture label, value is the linked dataset).
        :return: comparision results
        """
        # Check parameters
        if not isinstance(gesture_hmms, dict):
            raise ("gesture_hmms must be a dictionary of hidden markov models.")
        self.gesture_hmms = gesture_hmms
        if not isinstance(gesture_datasets, dict):
            raise ("dataset_dir must be a dictionary of CsvDataset objects.")
        self.gesture_datasets = gesture_datasets
        # comparison results
        self.result = Result()
        # comparasing gesture_hmms
        for gesture_label, gesture_dataset in self.gesture_datasets.items():
            # for each dataset's file
            for sequence in gesture_dataset.readDataset():
                self.__comparison(sequence=sequence, dataset_label=gesture_label)
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
        self.__createModel(gesture_expressions=gesture_expressions)
        # start comparison
        self.onlineTest(gesture_hmms=gesture_hmms, gesture_datasets=gesture_datasets)

    def onlineTest(self, gesture_hmms, gesture_datasets): pass

    ### Private methods ###
    def __createModel(self, gesture_expressions):
        """
            this method creates a hmm for each expression in gesture_expressions.
        :param gesture_expressions: a dictionary of deictic expression.
        :return: a dictionary of deictic hmms.
        """
        # Check parameters
        if not isinstance(gesture_expressions, dict):
            raise ("gesture_hmms must be a dictionary of deictic expressions.")
        gesture_hmms = dict
        # create hmms object from deictic expressiion
        for gesture_label, expressions in gesture_expressions.items():
            gesture_hmms[gesture_label] = []
            for expression in expressions:
                # create model
                model = None
                gesture_hmms[gesture_label].append(model)
        return gesture_hmms

    def __comparison(self, sequence, dataset_label):
        """
            given a sequence, this methods computes the log probability for each model.
        :param sequence: a sequence of frames.
        :param dataset_label: the list of gesture labesl.
        :return:
        """
        # Max probability
        max_norm_log_probability = -sys.maxsize
        # Compute log probability for each model
        for gesture_label, models in self.gesture_hmms.items():
            for model in models:
                # Computes sequence's log-probability and normalized
                log_probability = model.log_probability(sequence)
                norm_log_probability = log_probability / len(sequence)
            # Checks which is the best result
            if (norm_log_probability > max_norm_log_probability):
                max_norm_log_probability = norm_log_probability
                index_label = gesture_label
        # Update results
        self.result.update(row_label=dataset_label, column_label=index_label)
