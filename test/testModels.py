from dataset import *
import matplotlib.pyplot as plt
from dataset import *
import os
import csv
import numpy
import sys
from gesture import *

def wrong_test(model, wrong_dir, dimensions=2):
    wrong = DatasetIterator(wrong_dir)

    for filename in wrong.getCsvDataset():
        sequence = wrong.read_file(filename, dimensions, scale=100)
        print('{} log-probability: {}, normalised-log-probability {}'.format(
            filename, model.log_probability(sequence),
            model.log_probability(sequence) / len(sequence)
        ))

def compare_models_test(model_1, model_2, dir, dimensions = 2):
    dataset = DatasetIterator(dir)

    for filename in dataset.getCsvDataset():
        sequence = dataset.read_file(filename, dimensions, scale=100)
        print(model_1.name +' - {} log-probability: {}, normalised-log-probability {}'.format(
            filename, model_1.log_probability(sequence),
            model_1.log_probability(sequence) / len(sequence)
        ))
        print(model_2.name + ' - {} log-probability: {}, normalised-log-probability {}'.format(
            filename, model_2.log_probability(sequence),
            model_2.log_probability(sequence) / len(sequence)
        ))
        print()


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
        sequences = list_dataset[index_dataset].read_dataset()

        # Max probability, index gestureindex model
        max_norm_log_probability = -sys.maxsize
        index_model = -1

        # For each sequence
        for sequence in sequences:
            if plot:
                plt.plot(sequence[:, 0], sequence[:, 1], label=filename, marker='.')
                plt.title(list_dataset[index_dataset])
            # for each model
            for i in range(0, len(models)):
                if plot:
                    c = numpy.array(models[i].sample()).astype('float')
                    plt.plot(c[:, 0], c[:, 1], label=models[i].name, marker='.')

                # Computes sequence's log-probability and normalized
                log_probability = models[i].log_probability(sequence)
                norm_log_probability = log_probability / len(sequence)

                # Stampo i risultati
                #print('{} - {} log-probability: {}, normalised-log-probability {}'.format(index_file,
                #    models[i].name, log_probability, norm_log_probability))

                # Check which is the best result
                if(norm_log_probability > max_norm_log_probability):
                    max_norm_log_probability = norm_log_probability
                    index_model = i

            # Aggiorno matrice risultati
            results[index_dataset][index_model] += 1 #results[index_dataset][index_model] + 1

            if plot:
                plt.show()

    # Salva risultati
    print(results)
    #save_confusion_matrix(results, filename, names)
    return results

# Saves results into csv file
def save_confusion_matrix(results, filename, names):

    with open(filename,'wb') as file:
        # Headers row
       # numpy.savetxt(file, names,  delimiter=',', newline=" ", fmt='%s')

        # Data
        for i in range(0, len(results)):
            # Header col
            #numpy.savetxt(filename, names[i], delimiter=',', fmt='%s')
            # Results
            numpy.savetxt(filename, results[i], delimiter=',')

    # Send email
