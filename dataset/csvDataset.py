# csvDataset.py
# Authors: Alessandro Carcangiu, Davide Spano

"""
Class for reading and analysing data from a given dataset. All dataset must be converted in csv format.

"""
import csv
import numpy
import matplotlib.pyplot as plt
import os
from shutil import copyfile
# random
import datetime
import random
# HiddenMarkovModel
from pomegranate import HiddenMarkovModel


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
    def __init__(self, dir, type=float):
        """

        :param dir: the path of the directory wich contains the files.
        :param type: the type of data.
        """
        # Check parameters
        if not isinstance(dir, str):
            raise TypeError
        # Initialize parameters
        self.dir = dir
        self.type = type
        self.compositeTransform = CompositeTransform()

    def getDatasetIterator(self):
        """ Returns an iterator on all csv files in a dataset """
        return DatasetIterator(self.dir)

    def readDataset(self, d=False):
        """ Returns a list of sequences, containing the samples in each file of the dataset"

        Returns
        --------
        sequences : list
            the list of sequences and their filenames

        """
        sequences = [];
        i = 0
        for filename in self.getDatasetIterator():
            seq = self.readFile(filename=filename);
            # Creates tuple
            tuple = [seq, filename]
            # Add tuple in sequences
            sequences.append(tuple)
            if d:
                print("{0}: file {1}".format(i, filename))
            i += 1
        return sequences;

    def readFile(self, filename):
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
            if self.type == str:
                result = []
                for item in vals:
                    result.append(item[0])
            else:
                result = numpy.array(vals).astype(self.type)
        return result

    def addTransform(self, transform):
        self.compositeTransform.addTranform(transform)

    def applyTransforms(self, outputDir=None):
        sequences = []
        if not outputDir is None and not os.path.exists(outputDir):
            os.makedirs(outputDir)
        for file in self.getDatasetIterator():
            sequence = self.readFile(file)
            sequence = self.compositeTransform.transform(sequence)
            sequences.append([sequence, file])
            if not outputDir is None:
                numpy.savetxt(outputDir + file, sequence, delimiter=',')

        return sequences

    def crossValidation(self, iteration=0, x = 10):
        """
            Selects the 'x' part of the files in the dataset as test and uses the other ones for the training phase.

        :param x: specifies the portion of files to use in the test phase
        :return: the two created list
        """
        files = self.readDataset()
        train_files = []
        test_files = []
        num_test_files = int(len(files)/x)

        # Take test files from train list
        #random.seed(datetime.datetime.now())
        #for i in range(num_test_files):
        # generate nth unique indexes randomly, such that 0 <= index <= lenght of files
        #indexes = random.sample(range(0, len(files)), len(files))

        # push into test_files the file in position index and remove it from train_files
        indexes = [index for index in range(iteration*num_test_files, (iteration+1)*num_test_files)]

        for i in range(len(files)):
            if i in indexes:
                test_files.append(files[i])
            else:
                train_files.append(files[i])

        return train_files, test_files

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
    def plot(self, compared_dataset = None, compared_model = None, sampleName = None, singleMode = False, dimensions = 2):
        """

        :param dimensions:
        :param sampleName:
        :param model:
        :param singleMode:
        :return:
        """
        # Check and initilize parameters
        if compared_dataset != None and (not isinstance(compared_dataset, CsvDataset)):
            raise TypeError
        elif compared_dataset != None:
            compared_files = compared_dataset.getDatasetIterator().filenames
        if compared_model != None and not isinstance(compared_model, HiddenMarkovModel):
            raise TypeError

        # Intitialize 3d parameters
        fig = plt.figure(2);
        if dimensions == 3:
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
        else:
            plt.axis('equal')

        files = self.getDatasetIterator().filenames
        for filename in files:
            if sampleName == None or filename == sampleName:
                with open(self.dir + filename, "r") as f:
                    # get data
                    sequence = self.readFile(filename=filename)
                    sequence_plot = CsvDataset.__plot(sequence=sequence)
                    # Compares sample with a model or other dataset
                    if compared_model != None or compared_dataset != None:
                        # sample generated by model
                        if compared_model != None:
                            compared_sequence_plot = self.__plot(numpy.array(compared_model.sample()).astype('float'))
                        # file from other dataset (choice randomly)
                        if compared_dataset != None:
                            compared_file = compared_files[random.randint(len(compared_files))]
                            compared_sequence_plot = compared_dataset.readFile(filename=compared_file)
                        # Plot
                        plt.legend((sequence_plot[0], compared_sequence_plot[0]),
                                   ('original sequence', 'compared sequence'), loc='lower right')
            # Plot single file
            if singleMode or sampleName != None:
                # Initizalize axis
                plt.title(sampleName)
                plt.show()
        # Plot all files together
        if not singleMode:
            plt.show()

    ### Private methods ###
    @staticmethod
    def __plot(sequence, dimensions=2, filename = ""):
        """
            plots the given sequence and its points into the specified canvas.
        :param sequence: the sequence to plot.
        :param dimensions: x,y or x,y and z.
        :param filename: the name of the sequence.
        :return:
        """
        # Check parameters
        if not isinstance(sequence, numpy.ndarray):
            raise TypeError
        # Plot sequence
        if dimensions == 2: # 2 dimensions
            plt.scatter(sequence[:, 0], sequence[:, 1])
            for i in range(0, len(sequence)):
                plt.annotate(str(i), (sequence[i, 0], sequence[i, 1]))
            return plt.plot(sequence[:, 0], sequence[:, 1], label=filename)
        else: # 3 dimensions
            plt.scatter(sequence[:, 0], sequence[:, 1], sequence[:, 2])
            for i in range(0, len(sequence)):
                plt.annotate(str(i), (sequence[i, 0], sequence[i, 1], sequence[i, 2]))
            return plt.plot(sequence[:, 0], sequence[:, 1], sequence[:, 1], sequence[:, 2], label=filename)




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