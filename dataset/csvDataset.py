# csvDataset.py
# Authors: Alessandro Carcangiu, Davide Spano

"""
Class for reading and analysing data from a given dataset. All dataset must be converted in csv format.

"""
import csv
import numpy
import matplotlib.pyplot as plt
import os
from random import randint
from shutil import copyfile


class DatasetIterator:

    def __init__(self, dir, filter = '.csv'):
        self.filenames = os.listdir(dir)
        self.dirname = dir
        self.filter = filter
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.i < len(self.filenames):
            if self.filenames[self.i].endswith(self.filter):
                self.i += 1
                return self.filenames[self.i - 1]
            else:
                self.i+= 1
        raise StopIteration

class CsvDataset:
    """
    Class for dataset reading and plotting
    """
    def __init__(self, dir):
        self.dir = dir
        self.compositeTransform = CompositeTransform()

    def getDatasetIterator(self):
        """ Returns an iterator on all csv files in a dataset """
        return DatasetIterator(self.dir)

    def read_dataset(self, d=False):
        """ Returns a list of sequences, containing the samples in each file of the dataset"

        Returns
        --------
        sequences : list
            the list of sequences

        """
        sequences = [];
        i = 0
        for filename in self.getDatasetIterator():
            seq = self.read_file(filename);
            sequences.append(seq)
            if d:
                print("{0}: file {1}".format(i, filename))
            i += 1
        return sequences;

    def read_file(self, filename):
        """ Reads a single file, returning the samples in a list

        Parameters
        ----------
        filename: string
            the name of the file to read

        Returns
        --------
        result : list
            the list of samples

        """
        with open(self.dir + filename, "r") as f:
            reader = csv.reader(f, delimiter=',')
            vals = list(reader)
            result = numpy.array(vals).astype('float')
            return result

    def read_ten_cross_validation_dataset(self, inputDir, type, k = 0, model_index = None):
        files = open(inputDir+type+'_ten-cross-validation_{}.txt'.format(str(k))).readlines()
        files = files[0].split('/')
        sequences = []
        for filename in files:
            seq = self.read_file(filename);
            sequences.append(seq)
        return  sequences

    def addTransform(self, transform):
        self.compositeTransform.addTranform(transform)

    def applyTransforms(self, outputDir=None):
        sequences = []
        if not outputDir is None and not os.path.exists(outputDir):
            os.makedirs(outputDir)
        for file in self.getDatasetIterator():
            sequence = self.read_file(file)
            sequence = self.compositeTransform.transform(sequence)
            sequences.append(sequence)
            if not outputDir is None:
                numpy.savetxt(outputDir + file, sequence, delimiter=',')

        return sequences

    def ten_cross_validation(self, outputDir, k = 0, rates = None, labels = None):
        """ Selects the tenth part of the files in the dataset as test and uses the other ones for training

        Parameters
        ----------
        outputDir: str
            path where the list of files will be save
        k: int
        rates: list
        labels: list
        """

        testing_dataset = []
        if(rates != None and labels != None):
            length = int(len(labels)/10)
            # Test files
            for index in range(0, len(rates)):
                num_file_label = int((length * rates[index][1])/100)
                for i in range(0, num_file_label):
                    flag = False
                    index_for_testing = -1
                    while flag == False:
                        index_for_testing = randint(0, len(labels)-1)
                        if labels[index_for_testing][1] == index:
                            flag = True
                    file_test = labels.pop(index_for_testing)
                    testing_dataset.append(file_test[0])
            # Training files
            training_dataset = []
            for row in labels:
                training_dataset.append(row[0]+'#'+str(row[1]))
        else:
            # Take all files
            training_dataset = self.getDatasetIterator().filenames
            # Test files number (the tenth part of the files in the dataset)
            length = int(len(training_dataset)/10)
            for i in range(0, length):
                index_for_testing = randint(0, len(training_dataset)-1)
                testing_dataset.append(training_dataset.pop(index_for_testing))

        # Save test and training files in csv file
        with open(outputDir+'train_ten-cross-validation_{}.txt'.format(str(k)), mode='wt', encoding='utf-8') as myfile:
            myfile.write('/'.join(training_dataset))
        with open(outputDir+'test_ten-cross-validation_{}.txt'.format(str(k)), mode='wt', encoding='utf-8') as myfile:
            myfile.write('/'.join(testing_dataset))

    def leave_one_out(self, conditionFilename=None, leave_index = -1):
        """ Selects one of the files in the dataset as test and uses the other ones for training

        Parameters
        ----------
        conditionFilename: fun
            function for defining test and train set
        leave_index: int
            index of the test file
        """

        if(conditionFilename):
            # Two list
            training_list = []
            testing_list = []
            for filename in self.getDatasetIterator():
                if conditionFilename(filename):# If true append the file to testing list
                    testing_list.append(filename)
                else:
                    training_list.append(filename)# If false append the file to training list
        else:
            # Gets all files
            training_list = self.read_dataset()
            # Removes file at the specified index
            testing_list = training_list.pop(leave_index)

        return testing_list, training_list

    # Plot
    # Plots input dataset's files
    def plot(self, dimensions = 2, sampleName = None, singleMode = False):
        fig = plt.figure();
        labels =[];
        ax = None
        if dimensions == 3:
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
        else:
            plt.axis('equal')

        for filename in self.getDatasetIterator():
            if sampleName == None or filename == sampleName:
                with open(self.dir + filename, "r") as f:
                    reader = csv.reader(f, delimiter=',')
                    vals = list(reader)
                    result = numpy.array(vals).astype('float')
                    if dimensions == 3:
                        ax.plot(result[:, 0], result[:, 1], result[:, 2], label=filename)
                    else:

                        fig, ax = plt.subplots()
                        ax.scatter(result[:,0], result[:,1])
                        for i in range(0, len(result)):
                            ax.annotate(str(i), (result[i,0], result[i,1]))
                        plt.axis('equal')
                        plt.plot(result[:, 0], result[:, 1], label=filename, marker='.')

                    if singleMode:
                        plt.title(filename)
                        plt.show()
                        labels.append(filename + "," + input(filename +"->"));
        if dimensions == 3:
            ax.legend()
        else:
            plt.legend(loc='upper right')

        if sampleName != None:
            plt.title(sampleName)
        if not singleMode:
            plt.show()
        print(labels)


class DatasetTransform:

    def transform(self, sequence):
        return sequence

class CompositeTransform(DatasetTransform):

    def __init__(self):
        self.transforms = []


    def addTranform(self, transform):
        if isinstance(transform, DatasetTransform):
            self.transforms.append(transform)

    def transform(self, sequence):
        current = sequence
        for t in self.transforms:
            current = t.transform(current)
        return current