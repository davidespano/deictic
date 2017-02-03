from dataset import *
import matplotlib.pyplot as plt
from dataset import *
import os
import csv
import numpy
import sys
from gesture import *

def wrong_test(model, wrong_dir, dimensions=2):
    wrong = LeapDataset(wrong_dir)

    for filename in wrong.getCsvDataset():
        sequence = wrong.read_file(filename, dimensions, scale=100)
        print('{} log-probability: {}, normalised-log-probability {}'.format(
            filename, model.log_probability(sequence),
            model.log_probability(sequence) / len(sequence)
        ))

def compare_models_test(model_1, model_2, dir, dimensions = 2):
    dataset = ToolsDataset(dir)

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


# Senza primitive
def compare_all_models_test_without_primitive(models, baseDir, results, dimensions = 2, scale = 100, index = 0):
    # Matrice risultati
    #results = numpy.zeros((len(models), len(models)), dtype=numpy.int)
    index_gesture = -1

    # Prendi ogni gesture in models
    for model in models:
        index_gesture = index_gesture + 1
        dataset = ToolsDataset(baseDir + '/' + model.name + '/')

        # Aggiusta index
        index_correct = index
        if index >= len(dataset.getCsvDataset().filenames):
            index_correct = len(dataset.getCsvDataset().filenames) - 1
        # Prendi la sequenza
        correct = ToolsDataset(baseDir + '/' + model.name + '/')
        one, sequences = correct.leave_one_out_dataset(index_correct, dimensions=dimensions, scale=scale)

        # Matrice risultati
        max_norm_log_probability = -sys.maxsize
        index_model = -1

        # Per ogni modello
        for i in range(0, len(models)):
            # Calcola la log probability della sequenza e la sua normalizzata
            log_probability = models[i].log_probability(one)
            norm_log_probability = log_probability / len(one)
            # Determino qual è la gesture con la probabilità più alta
            if (norm_log_probability > max_norm_log_probability):
                max_norm_log_probability = norm_log_probability
                index_model = i

        # Aggiorno matrice risultati
        results[index_gesture][index_model] = results[index_gesture][index_model] + 1

    # Salva risultati
    save_confusion_matrix(results, baseDir, models, index=index)
    return  results

# Compara tutti i modelli con tutte le gesture definite
def compare_all_models_test(models, baseDir, dimensions = 2, scale = 100):
    # Matrice risultati
    results = numpy.zeros((len(models), len(models)), dtype=numpy.int)
    index_gesture = -1

    # Prendi ogni gesture in models
    for model in models:
        index_gesture = index_gesture + 1
        dataset = ToolsDataset(baseDir+model.name+'/')

        # Compara i modelli con ogni file
        for filename in dataset.getCsvDataset():
            # Prendi la sequenza
            sequence = dataset.read_file(filename, dimensions, scale)
            # Matrice risultati
            max_norm_log_probability = -sys.maxsize
            index_model = -1

            # Per ogni modello
            for i in range(0, len(models)):
                # Calcola la log probability della sequenza e la sua normalizzata
                log_probability = models[i].log_probability(sequence)
                norm_log_probability = log_probability / len(sequence)

                # Stampo i risultati
                #print('{} log-probability: {}, normalised-log-probability {}'.format(
                #    filename, log_probability, norm_log_probability))

                # Determino qual è la gesture con la probabilità più alta
                if(norm_log_probability > max_norm_log_probability):
                    max_norm_log_probability = norm_log_probability
                    index_model = i

            # Aggiorno matrice risultati
            results[index_gesture][index_model] = results[index_gesture][index_model] + 1

    # Salva risultati
    save_confusion_matrix(results, baseDir, models)
    return  results

# Salva risultati in un file csv
def save_confusion_matrix(results, baseDir, models, index=-1):
    sequences = []

    # Salva headers riga
    headers = ["" for x in range(len(models))]
    for i in range(0, len(models)):
        headers[i] = models[i].name
    numpy.savetxt(baseDir + '/apppend.csv', headers, fmt="%s")

    # Salva matrice risultati + header colonna
    for i in range(0, len(results)):
        sequences.append(results[i])#numpy.concatenate((headers[i], results[i]), axis=0))

    # Salva il tutto in un file
    if(index != -1):
        numpy.savetxt(baseDir + '/results_no_primitive.csv', sequences, fmt="%d")
    else:
        numpy.savetxt(baseDir + '/results.csv', sequences, fmt="%d")

# Verifica se una certa gesture debba essere valutata o no
def check_folder_model(folder, models):
    for model in models:
        if folder == model.name:
            return True

    return False