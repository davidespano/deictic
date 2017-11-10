import numpy
import csv
import matplotlib as plt
import itertools

class Result():
    """
        Results provides the methods for managing, plotting and saving confusion matrix.
    """

    def __init__(self, gesture_labels = []):
        # Check parameters
        if not isinstance(gesture_labels, list):
            raise ("name_gestures must be a list of object.")
        # gesture_labels
        self.labels = gesture_labels
        # array (the core of the confusion matrix)
        self.array = numpy.zeros((len(self.labels), len(self.labels)), dtype=int)

    def update(self, row, column):
        """
            this method autoincrements the specified index.
        :param row: the datesef from which the file came.
        :param coloumn: the gesture which recognized this file.
        :return:
        """
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


class Test():

    # singleton
    singleton = None

    ### Public methods ###
    @staticmethod
    def getInstance():
        if Test.singleton == None:
            singleton = Test()
        return singleton

    def offlineTest(self, gesture_hmms, gesture_datasets):
        """
            offlineTest starts the offline test.
        :param gesture_hmms: a dictionary of hidden markov models (key is the gesture label, values are his expression models).
        :param gesture_datasets: is a dictionary of CsvDataset objects (key is the gesture label, value is the linked dataset).
        :return: comparision results
        """
        # Check parameters
        if not isinstance(gesture_hmms, dict):
            raise ("hmms must be a dictionary of hidden markov models.")
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

    def onlineTest(self): pass

    ### Private methods ###
    def __comparison(self, sequence, dataset_label):
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
        self.result.update()
