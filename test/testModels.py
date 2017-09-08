from dataset import *
import csv
from gesture import *
# Event
from axel import Event
# Copy
import copy

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



def label_class(groups, baseDir, outputDir):

    # Get all gesture's dataset
    k = groups.keys()
    for name in k:
        name = name

    os.mkdir(outputDir + name + '/')
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
    def __init__(self, models, gesture_names, datasetDir=None,
                 plot_result = False, results=None):
        #super()
        if isinstance(models, list):
            # List of hmms
            self.models = models
        if isinstance(datasetDir, str):
            # Path which contained the datasets
            self.datasetDir = datasetDir
            # Gets gesture's testing dataset
            self.list_dataset = []
            for name in gesture_names:
                self.list_dataset.append(CsvDataset(datasetDir + name + '/'))
            # Namefile
            self.filename = self.datasetDir + 'matrix_confusion_choice.csv'
        if isinstance(gesture_names, list):
            # The list of gesture to recognized
            self.gesture_names = gesture_names
        if isinstance(plot_result, bool):
            # Plot results
            self.plot_result = plot_result
        if isinstance(results, numpy.ndarray):
            # The developer pass an older of results
            self.results = results
        else:
            self.results = numpy.zeros((len(models), len(models)), dtype=numpy.int)


    # Test models with all dataset's files
    def all_files(self):
        for index_dataset in range(0, len(self.list_dataset)):
            print("gesture {0}:".format(self.list_dataset[index_dataset].dir))

            # Gets sequences
            sequences = self.list_dataset[index_dataset].readDataset()
            # Compares models
            self.compares_models(sequences, index_dataset)
        return self.results
    # Test models with ten cross validation (using dataset's files)
    def ten_cross_validation(self, list_filesDir, k=0):
        self.results = numpy.zeros((len(self.models), len(self.models)), dtype=numpy.int)
        for index_dataset in range(0, len(self.list_dataset)):
            print("gesture {0}:".format(self.list_dataset[index_dataset].dir))

            # Gets sequences
            sequences = self.list_dataset[index_dataset].\
                read_ten_cross_validation_dataset(list_filesDir+self.gesture_names[index_dataset]+'/','test', k)
            # Compares models
            self.compares_models(sequences, index_dataset)
        return self.results
    # Test models with a single file
    def single_file(self, sequences):
        self.results = []
        # Compares models
        self.__compares_models(sequences)
        return self.results


    # Compares models with dataset
    def compares_models(self, sequences, index_dataset = None):
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
                    models[model].name, log_probability, norm_log_probability))

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
    def __compares_models(self, sequences):
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





class testRealTime():
    """
        This class implements the methods for testing deictic in real time
    """
    def __init__(self, hmms, plot_result = False):
        # Hmms
        self.hmms = hmms
        # User option regarding the plotting of the computation
        self.plot_result = plot_result
        #### Results ####
        # - log_probabilities: for each hmm this dictionary reports its norm log-probability
        # - frame_log_probabilities: for each frame of a file it reports the high probabilities for each hmm        self.log_probabilities = {}
        self.log_probabilities = {}
        for hmm in hmms:
            self.log_probabilities[hmm.name] = []
        self.frame_log_probabilities = []
        #self.results_for_file = []
        #### Events ####
        # Is raised when the system completed to fire a file, it is used for sending the results.
        self.managing_frame = Event()
        self.updateResults = Event()

    def computeLogProbability(self, frame, buffer):
        """
            passes the content of buffer to all hmms and returns the norm log-probability of each one.
        :param frame: the frame received
        :param buffer: the list of the latest sent frames
        :return: list of norm log-probabilities
        """
        for hmm in self.hmms:
            # Computes sequence's log-probability and its normalize
            log_probability = hmm.log_probability(buffer)
            norm_log_probability = log_probability / len(buffer)

            # Print debug results
            if(plot_gesture == True):
                print('Model:{} log-probability: {}, normalised-log-probability {}'.
                      format(hmm.name, log_probability, norm_log_probability))

            # Update log_probabilities
            self.log_probabilities[hmm.name].append(norm_log_probability)
            # Update log_probabilities_for_frame
            self.frame_log_probabilities.append([hmm.name, norm_log_probability])


        # new_item: coordinates points(x and y), stroke number, log temp and the hmm with the higher norm log-probability value and its value
        # max_log_probability=-sys.maxsize
        # best_hmm = self.hmms[0].name
        # for hmm in self.hmms:
        #     value = self.log_probabilities[hmm.name][-1]
        #     if value > max_log_probability:
        #         max_log_probability = value
        #         best_hmm = hmm.name
        # new_item = [frame[0], frame[1], frame[2], frame[3], best_hmm]
        # self.results_for_file.append(new_item)
        # Notifies that the new frame is managed
        self.managing_frame(copy.deepcopy(self.frame_log_probabilities))
        self.frame_log_probabilities.clear()

    def compareClassifiers(self, filename):
        """
            compares the classifiers in order to find the best hmm and updates the dictionary "result"
        :return:
        """
        # Notifies updating
        self.updateResults(copy.deepcopy(self.log_probabilities),
                           filename)
        # Clears data structures
        for hmm in self.hmms:
            self.log_probabilities[hmm.name] = []

