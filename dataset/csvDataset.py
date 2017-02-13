# csvDataset.py
# Authors: Alessandro Carcangiu, Davide Spano

"""
Class for reading and analysing data from a given dataset. All dataset must be converted in csv format.

"""
import csv
import numpy
import matplotlib.pyplot as plt
import os


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

    def read_dataset(self):
        """ Returns a list of sequences, containing the samples in each file of the dataset"

        Returns
        --------
        sequences : list
            the list of sequences

        """
        sequences = [];
        for filename in self.getDatasetIterator():
            seq = self.read_file(filename);
            sequences.append(seq)
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

    def addTransform(self, transform):
        self.compositeTransform.addTranform(transform)

    def applyTransforms(self, outputDir):
        if not os.path.exists(outputDir ):
            os.makedirs(outputDir)
        for file in self.getDatasetIterator():
            sequence = self.read_file(file)
            sequence = self.compositeTransform.transform(sequence)
            numpy.savetxt(outputDir + file, sequence, delimiter=',')


    def leave_one_out(self, leave_index = 0):
        """ Selects one of the files in the dataset as test and uses the other ones for training

        Parameters
        ----------
        leave_index: int
            index of the test file
        """

        # Gets all files
        sequences = self.read_dataset()
        # Removes file at the specified index
        leave = sequences.pop(leave_index)

        return leave, sequences

    # Plot
    # Plots input dataset's files
    def plot(self, dimensions = 2, sampleName = None):
        fig = plt.figure();
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
                        plt.plot(result[:, 0], result[:, 1], label=filename, marker='.')
        if dimensions == 3:
            ax.legend()
        else:
            plt.legend(loc='upper right')

        if sampleName != None:
            plt.title(sampleName)
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