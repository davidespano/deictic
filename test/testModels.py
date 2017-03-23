from dataset import *
import matplotlib.pyplot as plt
from dataset import *
import os
import csv
import numpy
import sys
from gesture import *

## Compare adhoc HMM
#
def compares_adhoc_models(models, sequences, gestureDir, results, dimensions = 2):
    # Namefile
    filename = gestureDir+'adhoc-hmm_results.csv'

    # Get all gesture's dataset
    list_dataset = []
    for model in models:
        list_dataset.append(CsvDataset(gestureDir+model.name+'/'))

    index_gesture = 0
    # For each gesture's test sequence
    for sequence in sequences:

        # Max probability, index gestureindex model
        max_norm_log_probability = -sys.maxsize
        index_model = -1

        # Prendi ogni gesture in models
        for i in range(0, len(models)):

            # Per ogni modello
            # Calcola la log probability della sequenza e la sua normalizzata
            log_probability = models[i].log_probability(sequence)
            norm_log_probability = log_probability / len(sequence)
            # Determino qual è la gesture con la probabilità più alta
            if (norm_log_probability > max_norm_log_probability):
                max_norm_log_probability = norm_log_probability
                index_model = i

        # Aggiorno matrice risultati
        results[index_gesture][index_model] = results[index_gesture][index_model] + 1
        index_gesture = index_gesture + 1

    # Salva risultati
    save_confusion_matrix(results, filename, models)
    return  results

## Compare deictic model
# Compara tutti i modelli con tutte le gesture definite
def compares_deictic_models(models, baseDir, names, plot=False):

    # Namefile
    filename = baseDir+'deictic_results.csv'

    # Confusion Matrix (n * n, where n is the number of models)
    results = numpy.zeros((len(models), len(models)), dtype=numpy.int)

    # Get all gesture's dataset
    list_dataset = []
    for name in names:
        list_dataset.append(CsvDataset(baseDir+name+'/'))

    # For each gesture's dataset
    for index_dataset in range(0, len(list_dataset)):

        print("gesture {0}: {1}".format(index_dataset, list_dataset[index_dataset].dir))

        # Get all sequence files
        sequences = list_dataset[index_dataset].read_dataset(d=False)

        # For each sequence
        j = 0
        for sequence in sequences:
            # Max probability, index gestureindex model
            max_norm_log_probability = -sys.maxsize
            index_model = -1
            # for each model
            for i in range(0, len(models)):

                # Computes sequence's log-probability and normalized
                log_probability = models[i].log_probability(sequence)
                norm_log_probability = log_probability / len(sequence)

                #print("file {0}: {1}".format( j,  norm_log_probability))

                # Check which is the best result
                if(norm_log_probability > max_norm_log_probability):

                   # print("change index: old {0} (p={1}); new {2} (p={3})".format(
                   #     index_model, max_norm_log_probability, i, norm_log_probability))
                    max_norm_log_probability = norm_log_probability
                    index_model = i

            #if index_model != index_dataset:
            #    print("file {0} not recognized".format(j))
            j +=1
            # Aggiorno matrice risultati
            results[index_dataset][index_model] += 1 #results[index_dataset][index_model] + 1

    return results

def compares_deictic_models2(groups, baseDir,  plot=False):
    filename = baseDir + 'deictic_results.csv'

    # Confusion Matrix (n * n, where n is the number of models)
    results = numpy.zeros((len(groups.keys()), len(groups.keys())), dtype=numpy.int)

    # Get all gesture's dataset
    list_dataset = []
    for name in groups.keys():
        list_dataset.append(CsvDataset(baseDir + name + '/'))

    # For each gesture's dataset
    for index_dataset in range(0, len(list_dataset)):
        print("gesture {0}: {1}".format(index_dataset, list_dataset[index_dataset].dir))
        # Get all sequence files
        sequences = list_dataset[index_dataset].read_dataset(d=False)

        # For each sequence
        j = 0
        for sequence in sequences:
            # Max probability, index gestureindex model
            max_norm_log_probability = -sys.maxsize
            index_model = -1
            # for each group

            i = 0
            for k in groups.keys():
                group = groups[k]

                max_group = -sys.maxsize

                for model in group:
                    log_probability = model.log_probability(sequence)
                    norm_log_probability = log_probability / len(sequence)

                    if(norm_log_probability > max_group):
                        max_group = norm_log_probability

                if (max_group > max_norm_log_probability):
                    # print("change index: old {0} (p={1}); new {2} (p={3})".format(
                    #     index_model, max_norm_log_probability, i, norm_log_probability))
                    max_norm_log_probability = max_group
                    index_model = i
                i += 1
            #if index_model != index_dataset:
            #    print("file {0} not recognized".format(j))
            j += 1
            # Aggiorno matrice risultati
            results[index_dataset][index_model] += 1  # results[index_dataset][index_model] + 1

    return results

## Compare deictic model
class test:

    def __init__(self, models, datasetDir, gesture_names, plot=False, results=None):
        super()
        if isinstance(models, list):
            self.models = models
        if isinstance(datasetDir, str):
            self.datasetDir = datasetDir
        if isinstance(gesture_names, list):
            self.gesture_names = gesture_names
        if isinstance(plot, bool):
            self.plot = plot
        if isinstance(results, numpy.ndarray):
            self.results = results
        else:
            self.results = numpy.zeros((len(models), len(models)), dtype=numpy.int)
        # Gets gesture's testing dataset
        self.list_dataset = []
        for name in gesture_names:
            self.list_dataset.append(CsvDataset(datasetDir+name+'/'))
        # Namefile
        self.filename = self.datasetDir+'deictic_results.csv'

    def all_files(self):
        for index_dataset in range(0, len(self.list_dataset)):
            print("gesture {0}:".format(self.list_dataset[index_dataset].dir))

            # Gets sequences
            sequences = self.list_dataset[index_dataset].read_dataset()
            # Compares models
            self.compares_models(sequences, index_dataset)

        return self.results

    def ten_cross_validation(self, list_filesDir, iterations=10):
        for i in range(0, iterations):
            for index_dataset in range(0, len(self.list_dataset)):
                print("gesture {0}:".format(self.list_dataset[index_dataset].dir))

                # Gets sequences
                sequences = self.list_dataset[index_dataset].read_ten_cross_validation_dataset(list_filesDir+'/{}/'.format(i+1)+
                                                                                               self.gesture_names[index_dataset]+'/', 'test')
                # Compares models
                self.compares_models(sequences, index_dataset)

        return self.results

    def compares_models(self, sequences, index_dataset):

        # For each sequence
        for sequence in sequences:

            # Max probability, index gestureindex model
            max_norm_log_probability = -sys.maxsize
            index_model = -1

            if self.plot:
                plt.plot(sequence[:, 0], sequence[:, 1], label=filename, marker='.')
                plt.title(list_dataset[index_dataset])

            # for each model
            for i in range(0, len(self.models)):
                if self.plot:
                    c = numpy.array(self.models[i].sample()).astype('float')
                    plt.plot(c[:, 0], c[:, 1], label=self.models[i].name, marker='.')

                # Computes sequence's log-probability and normalized
                log_probability = self.models[i].log_probability(sequence)
                norm_log_probability = log_probability / len(sequence)

                # Stampo i risultati
                #print('{} - {} log-probability: {}, normalised-log-probability {}'.format(index_file,
                #    models[i].name, log_probability, norm_log_probability))

                # Check which is the best result
                if(norm_log_probability > max_norm_log_probability):
                    max_norm_log_probability = norm_log_probability
                    index_model = i

            # Aggiorno matrice risultati
            self.results[index_dataset][index_model] += 1

            if self.plot:
                plt.show()

        # Salva risultati
        self.save_confusion_matrix()

    # Saves results into csv file
    def save_confusion_matrix(self):

        # Results
        numpy.savetxt(self.filename, self.results, delimiter=',')

        size = len(self.gesture_names)+1
        # Char matrix for results
        results_string = []
        # Headers
        headers = []
        headers.append('models')
        for i in range(1,size):
            headers.append(self.gesture_names[i-1])
        results_string.append(headers)
        # Values
        for i in range(0, size-1):
            new_row = []
            new_row.append(self.gesture_names[i-1])
            for j in range(0,size-1):
                new_row.append(str(self.results[i,j]))
            results_string.append(new_row)

        with open('matrix_confusion.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in results_string:
                spamwriter.writerow(row)