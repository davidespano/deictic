from dataset import *
import csv
from gesture import *
# Event
from axel import Event
# Copy
import copy


# todo: delete #
## Compare deictic model
def compares_deictic_models(groups, baseDir, ten_fold = False, fold =0):
    # Filename results
    filename = baseDir + 'matrix_confusion.csv'
    # Gesture names
    names = []
    for name in groups.keys():
        names.append(name)

    # Confusion Matrix (n * n, where n is the number of models)
    results = numpy.zeros((len(groups.keys()), len(groups.keys())), dtype=numpy.int)

    # Get all gesture's dataset
    list_dataset = []
    for name in groups.keys():
        if not ten_fold:
            list_dataset.append(CsvDataset(baseDir + name + '/'))
        else:
            files = open(baseDir + "../ten-cross-validation/" + name  + '/test_ten-cross-validation_{}.txt'.format(str(fold))).readlines()
            files = files[0].split('/')
            list_dataset.append([name, files])

    # For each gesture's dataset
    for index_dataset in range(0, len(list_dataset)):
        if not ten_fold:
            dir = list_dataset[index_dataset].dir
        else:
            dir = list_dataset[index_dataset][0]
        print("gesture {0}: {1}".format(index_dataset, dir ))
        # Get all sequence files
        if not ten_fold:
            sequences = list_dataset[index_dataset].readDataset(d=False)
        else:
            sequences = []
            for el in list_dataset[index_dataset][1]:
                with open("{0}{1}/{2}".format(baseDir, list_dataset[index_dataset][0], el), "r") as f:
                    reader = csv.reader(f, delimiter=',')
                    vals = list(reader)
                    sequence = numpy.array(vals).astype('float')
                    sequences.append(sequence)

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

        # Salva risultati
        size = len(names)+1
        # Char matrix for results
        results_string = []
        # Headers
        headers = []
        headers.append('models')
        for i in range(1,size):
            headers.append(names[i-1])
        results_string.append(headers)
        # Values
        for i in range(0, size-1):
            new_row = []
            new_row.append(names[i])
            for j in range(0,size-1):
                new_row.append(str(results[i,j]))
            results_string.append(new_row)

        with open(filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in results_string:
                spamwriter.writerow(row)

    return results



def test_dict(hmms, baseDir):
    results = numpy.zeros((len(hmms), len(hmms)), dtype=numpy.int)
    index_dataset = 0
    for name_gesture in hmms:
        dataset = CsvDataset(baseDir+name_gesture+"/").readDataset()
        for sequence in dataset:
            data = sequence[0]
            # Max probability, index gesture-index model
            max_norm_log_probability = -sys.maxsize
            index_model = -1

            # for each model
            index = -1
            for key in hmms:
                index+=1
                for hmm in hmms[key]:
                    # Computes sequence's log-probability and normalized
                    log_probability = hmm.log_probability(data)
                    norm_log_probability = log_probability / len(data)
                    # Checks which is the best result
                    if (norm_log_probability > max_norm_log_probability):
                        max_norm_log_probability = norm_log_probability
                        index_model = index
            # Update array result
            results[index_dataset][index_model] += 1
        # update index_dataset - index_model
        index_dataset+=1

    # Save results
    # Results
    size = len(hmms) + 1
    # Char matrix for results
    results_string = []
    # Headers
    headers = []
    headers.append('models')
    for name_gesture in hmms:
        headers.append(name_gesture)
    results_string.append(headers)
    # Values
    i = -1
    for name_gesture in hmms:
        i+=1
        new_row = []
        new_row.append(name_gesture)
        for j in range(0, size - 1):
            new_row.append(str(results[i, j]))
        results_string.append(new_row)

    with open(baseDir+"matrix_confusion_choice.csv", 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in results_string:
            spamwriter.writerow(row)


def label_class(groups, baseDir, outputDir):

    # Get all gesture's dataset
    k = groups.keys()
    for name in k:
        name = name

    #os.mkdir(outputDir + name + '/')
    dataset = CsvDataset(baseDir + name + '/')
    group = groups[name]

    report_string = []
    # For each file
    for filename in dataset.getDatasetIterator():
        # sequence
        sequence = dataset.readFile(filename)

        # Max probability, index gestureindex model
        max_norm_log_probability = -sys.maxsize
        index_model = -1

        new_row = []
        i = 0# counter model

        for model in group:
            log_probability = model.log_probability(sequence)
            norm_log_probability = log_probability / len(sequence)

            if (norm_log_probability > max_norm_log_probability):
                max_norm_log_probability = norm_log_probability
                index_model = i
            i += 1

        new_row.append(filename)
        new_row.append(str(index_model))
        report_string.append(new_row)


    with open(outputDir+name+'/report.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in report_string:
            spamwriter.writerow(row)




## Compare deictic model
class test:
    """This class implements the methods for comparing hmms"""
    def __init__(self, models, gesture_names, datasetDir, plot_result=False, results=None, file_path_results=None):
        #super()
        if isinstance(models, list):
            # List of hmms
            self.models = models
        if isinstance(datasetDir, str):
            # Path which contain the datasets
            self.datasetDir = datasetDir
            # Gets gesture's testing dataset
            self.list_dataset = []
            for name in gesture_names:
                self.list_dataset.append(CsvDataset(datasetDir + name + '/'))
        # path file results
        if isinstance(file_path_results, str):
            self.filename = file_path_results + 'matrix_confusion_choice.csv'
        else:
            self.filename = self.datasetDir + 'matrix_confusion_choice.csv'
        if isinstance(gesture_names, list):
            # The list of gesture to recognize
            self.gesture_names = gesture_names
        if isinstance(plot_result, bool):
            # Plot results? Yes or not
            self.plot_result = plot_result
        if isinstance(results, numpy.ndarray):
            # The developer passes older results
            self.results = results
        else:
            self.results = numpy.zeros((len(models), len(models)), dtype=numpy.int)


    def all_files(self):
        """
            Test models with all dataset's files
        :return:
        """
        for index_dataset in range(0, len(self.list_dataset)):
            print("gesture {0}:".format(self.list_dataset[index_dataset].dir))

            # Gets sequences
            sequences = self.list_dataset[index_dataset].readDataset()
            # Compares models
            self.__compares_models(sequences, index_dataset)
        return self.results
    def ten_cross_validation(self, list_filesDir, k=0):
        """
            test models with ten cross validation (using dataset's files)
        :param list_filesDir:
        :param k:
        :return:
        """
        self.results = numpy.zeros((len(self.models), len(self.models)), dtype=numpy.int)
        for index_dataset in range(0, len(self.list_dataset)):
            print("gesture {0}:".format(self.list_dataset[index_dataset].dir))

            # Gets sequences
            sequences = self.list_dataset[index_dataset].\
                read_ten_cross_validation_dataset(list_filesDir+self.gesture_names[index_dataset]+'/','test_real_time', k)
            # Compares models
            self.__compares_models(sequences, index_dataset)
        return self.results
    def single_file(self, sequences):
        """
            test models with a single file
        :param sequences:
        :return:
        """
        self.results = []
        # Compares models
        self.__compares_models_single_file(sequences)
        return self.results


    # Compares models with dataset
    def __compares_models(self, sequences, index_dataset = None):
        # Sequences is a list of tuples (data_features_movements and its filename)
        for tuple in sequences:
            # Gets sequence data
            sequence = tuple[0]
            # Max probability, index gesture-index model
            max_norm_log_probability = -sys.maxsize
            index_model = -1

            # for each model
            for model in range(0, len(self.models)):
                # Computes sequence's log-probability and normalized
                log_probability = self.models[model].log_probability(sequence)
                norm_log_probability = log_probability / len(sequence)

                # Print debug results
                if self.plot_result != False:
                    print('File:{} - Model:{} log-probability: {}, normalised-log-probability {}'.format(tuple[1],
                    self.models[model].name, log_probability, norm_log_probability))

                # Checks which is the best result
                if(norm_log_probability > max_norm_log_probability):
                    max_norm_log_probability = norm_log_probability
                    index_model = model
            # Update array result
            self.results[index_dataset][index_model] += 1

            # Shows recognition result
            if self.plot_result != False:
                print("Sequence: " + tuple[1] + " Model: " + self.gesture_names[index_model])
        # Saves results
        self.__save_confusion_matrix()
    # Compares models with a single file
    def __compares_models_single_file(self, sequences):
        # For each model
        for model in range(0, len(self.models)):
            # Computes sequence's log-probability and normalized
            log_probability = self.models[model].log_probability(sequences[model])#sequences[model])
            norm_log_probability = log_probability / len(sequences[model])#sequences[model])
            # Saves gesture name and norm log probability result
            res = []
            res.append(self.gesture_names[model])
            res.append(str(norm_log_probability))
            self.results.append(res)
            # Take minor

    # Saves results into csv file
    def __save_confusion_matrix(self):

        # Results
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
            new_row.append(self.gesture_names[i])
            for j in range(0,size-1):
                new_row.append(str(self.results[i,j]))
            results_string.append(new_row)

        with open(self.filename, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in results_string:
                spamwriter.writerow(row)

