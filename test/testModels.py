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
            # Aggiusta index
            #index_correct = index
            #if index >= len(dataset.getCsvDataset().filenames):
            #index_correct = len(dataset.getCsvDataset().filenames) - 1

            # Prendi la sequenza
            #correct = DatasetIterator(baseDir + '/' + model.name + '/')
            #one, sequences = correct.leave_one_out_dataset(index_correct, dimensions=dimensions, scale=scale)

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
def compares_deictic_models(models, baseDir, names):
    # Namefile
    filename = baseDir+'deictic_results.csv'

    # Confusion Matrix (n * n, where n is the number of models)
    results = numpy.zeros((len(models), len(models)), dtype=numpy.int)

    # Get all gesture's dataset
    list_dataset = []
    for name in names:
        list_dataset.append(CsvDataset(baseDir+name+'/'))
    #index_file = 0

    # For each gesture's dataset
    for index_dataset in range(0, len(list_dataset)):
        # Get all sequence files
        sequences = list_dataset[index_dataset].read_dataset()

        # Max probability, index gestureindex model
        max_norm_log_probability = -sys.maxsize
        index_model = -1

        # For each sequence
        for sequence in sequences:
            #print(' ')
            #index_file = index_file+1

            plt.plot(sequence[:, 0], sequence[:, 1], label=filename, marker='.')
            plt.title(list_dataset[index_dataset])
            # for each model
            for i in range(0, len(models)):
                c = numpy.array(models[i].sample()).astype('float')
                plt.plot(c[:, 0], c[:, 1], label=models[i].name, marker='.')

                # Calcola la log probability della sequenza e la sua normalizzata
                log_probability = models[i].log_probability(sequence)
                norm_log_probability = log_probability / len(sequence)

                # Stampo i risultati
                #print('{} - {} log-probability: {}, normalised-log-probability {}'.format(index_file,
                #    models[i].name, log_probability, norm_log_probability))

                # Determino qual è la gesture con la probabilità più alta
                if(norm_log_probability > max_norm_log_probability):
                    max_norm_log_probability = norm_log_probability
                    index_model = i

            # Aggiorno matrice risultati
            results[index_dataset][index_model] = results[index_dataset][index_model] + 1

            plt.show()

    # Salva risultati
    save_confusion_matrix(results, filename, models)
    return results

# Salva risultati in un file csv
def save_confusion_matrix(results, filename, models):
    sequences = []

    with open(filename,'wb') as file:
#        np.savetxt(f,x,fmt='%.5f')
        ### Save results ###
        i = models[0].name
        x = [model.name for model in models]
        # Headers row
        numpy.savetxt(file, x, delimiter=',', fmt='%s')

        # Data
        for i in range(0, len(results)):
            # Header col
            #numpy.savetxt(filename, models[i].name, delimiter=',')
            # Results
            numpy.savetxt(filename, results[i], delimiter=',')




    # Salva headers riga
    #headers = ["" for x in range(len(models))]
    #for i in range(0, len(models)):
    #    headers[i] = models[i].name
    #numpy.savetxt(baseDir + '/deictic_results.csv', headers, fmt="%s")

    # Salva matrice risultati + header colonna
    #for i in range(0, len(results)):
    #    sequences.append(results[i])#numpy.concatenate((headers[i], results[i]), axis=0))
    # Salva il tutto in un file
    #if(index != -1):
    #    numpy.savetxt(baseDir + '/adhoc_hmms.csv_results', sequences, fmt="%d")
    #else:
    #    numpy.savetxt(baseDir + '/deictic_results.csv', sequences, fmt="%d")

# Verifica se una certa gesture debba essere valutata o no
def check_folder_model(folder, models):
    for model in models:
        if folder == model.name:
            return True

    return False