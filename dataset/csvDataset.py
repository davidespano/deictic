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
# Lib for events like c#
import axel
# Timer delay (for simulating real time)
from time import sleep

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

    def readDataset(self, d=False, type=float):
        """ Returns a list of sequences, containing the samples in each file of the dataset"

        Returns
        --------
        sequences : list
            the list of sequences and their filenames

        """
        sequences = [];
        i = 0
        for filename in self.getDatasetIterator():
            seq = self.readFile(filename=filename, type=type);
            # Creates tuple
            tuple = [seq, filename]
            # Add tuple in sequences
            sequences.append(tuple)
            if d:
                print("{0}: file {1}".format(i, filename))
            i += 1
        return sequences;

    def readFile(self, filename, type=float):
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
            result = numpy.array(vals).astype(type)
            return result

    def read_ten_cross_validation_dataset(self, inputDir, type, k = 0, model_index = None):
        files = open(inputDir+type+'_ten-cross-validation_{}.txt'.format(str(k))).readlines()
        files = files[0].split('/')
        sequences = []
        for filename in files:
            seq = self.readFile(filename);
            sequences.append(seq)
        return  sequences

    def addTransform(self, transform):
        self.compositeTransform.addTranform(transform)

    def applyTransforms(self, outputDir=None):
        sequences = []
        if not outputDir is None and not os.path.exists(outputDir):
            os.makedirs(outputDir)
        for file in self.getDatasetIterator():
            sequence = self.readFile(file)
            sequence = self.compositeTransform.transform(sequence)
            sequences.append(sequence)
            if not outputDir is None:
                numpy.savetxt(outputDir + file, sequence, delimiter=',')

        return sequences

    def ten_cross_validation(self, outputDir, k = 0, rates = None, labels = None):
        """ Selects the tenth part of the files in the dataset as test_real_time and uses the other ones for training

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
            # test_real_time files
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
            # test_real_time files number (the tenth part of the files in the dataset)
            length = int(len(training_dataset)/10)
            for i in range(0, length):
                index_for_testing = randint(0, len(training_dataset)-1)
                testing_dataset.append(training_dataset.pop(index_for_testing))

        # Save test_real_time and training files in csv file
        with open(outputDir+'train_ten-cross-validation_{}.txt'.format(str(k)), mode='wt', encoding='utf-8') as myfile:
            myfile.write('/'.join(training_dataset))
        with open(outputDir+'test_ten-cross-validation_{}.txt'.format(str(k)), mode='wt', encoding='utf-8') as myfile:
            myfile.write('/'.join(testing_dataset))

    def leave_one_out(self, conditionFilename=None, leave_index = -1):
        """ Selects one of the files in the dataset as test_real_time and uses the other ones for training

        Parameters
        ----------
        conditionFilename: fun
            function for defining test_real_time and train set
        leave_index: int
            index of the test_real_time file
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
            training_list = self.readDataset()
            # Removes file at the specified index
            testing_list = training_list.pop(leave_index)

        return testing_list, training_list

    # Plot
    # Plots input dataset's files
    def plot(self, dimensions = 2, sampleName = None, model = None, singleMode = False):
        fig = plt.figure(2);
        ax = None
        if dimensions == 3:
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
        else:
            plt.axis('equal')

        files = self.getDatasetIterator().filenames
        for filename in files:
            if sampleName == None or filename == sampleName:
                with open(self.dir + filename, "r") as f:
                    reader = csv.reader(f, delimiter=',')
                    vals = list(reader)
                    result = numpy.array(vals).astype('float')

                    if dimensions == 3:
                        ax.plot(result[:, 0], result[:, 1], result[:, 2], label=filename)
                        plt.axis('equal')

                    else:
                        ax.scatter(result[:,0], result[:,1])
                        for i in range(0, len(result)):
                            ax.annotate(str(i), (result[i,0], result[i,1]))
                        plt.axis('equal')
                        original_sequence = plt.plot(result[:, 0], result[:, 1])
                        # Model
                        if model != None:
                            sequence = numpy.array(model.sample()).astype('float')
                            plt.axis("equal")
                            generated_sequece = plt.plot(sequence[:, 0], sequence[:, 1])
                            plt.legend((original_sequence[0], generated_sequece[0]),
                                   ('original sequence', 'generated sequence'), loc='lower right')
                        # Title
                        plt.title(filename)
            if singleMode:
                plt.show()

        if sampleName != None:
            plt.title(sampleName)
        if not singleMode:
            plt.show()




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