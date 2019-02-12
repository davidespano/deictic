# csvDataset.py
# Authors: Alessandro Carcangiu, Davide Spano

"""
Class for reading and analysing data from a given dataset. All dataset must be converted in csv format.

"""
import csv
import numpy
import matplotlib.pyplot as plt
import os
import copy
from shutil import copyfile
# random
import random
# HiddenMarkovModel
from pomegranate import HiddenMarkovModel
import ast

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

    def addTransforms(self, transforms):
        # check
        if not isinstance(transforms, list):
            raise TypeError
        for transform in transforms:
            self.addTransform(transform)

    def applyTransforms(self, output_dir=None):
        sequences = []
        if not output_dir is None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for file in self.getDatasetIterator():
            sequence = self.readFile(file)
            sequence = self.compositeTransform.transform(sequence)
            sequences.append([sequence, file])
            if not output_dir is None:
                # todo: manage strings or floats
                numpy.savetxt(output_dir + file, sequence, delimiter=',')
                #file = open(output_dir+file, 'w')
                #for item in sequence:
                #    file.write(item + "\n")

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
        # generate nth unique indices randomly, such that 0 <= index <= lenght of files
        #indices = random.sample(range(0, len(files)), len(files))

        # push into test_files the file in position index and remove it from train_files
        indices = [index for index in range(iteration*num_test_files, (iteration+1)*num_test_files)]

        for i in range(len(files)):
            if i in indices:
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
        if dimensions == 3:
            ax = plt.gca(projection='3d')
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
                            compared_name = compared_model.name
                            compared_sequence_plot = CsvDataset.__plot(numpy.array(compared_model.sample()).astype('float'))
                        # file from other dataset (choice randomly)
                        if compared_dataset != None:
                            compared_name = compared_dataset.dir
                            compared_file = compared_files[random.randint(0,len(compared_files)-1)]
                            compared_sequence = compared_dataset.readFile(filename=compared_file)
                            compared_sequence_plot = CsvDataset.__plot(compared_sequence)
                        # Plot
                        plt.legend((sequence_plot[0], compared_sequence_plot[0]),
                                   (self.dir, compared_name), loc='lower right')
                # Plot single file
                if singleMode or sampleName != None:
                    # Initizalize axis
                    plt.title(filename)
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
            return plt.plot(sequence[:, 0], sequence[:, 1], sequence[:, 2], label=filename)

class CsvDatasetExtended(CsvDataset):

    def __init__(self, dir, type=float):
        """
        :param dir: the path of the directory wich contains the files.
        :param type: the type of data.
        """
        # Check parameters
        if not isinstance(dir, str):
            raise TypeError
        if not os.path.isdir(dir):
            raise NotADirectoryError
        # Initialize parameters
        super(CsvDatasetExtended,self).__init__(dir, type)

    def readFile(self, filename):
        """

        :param filename:
        :return:
        """
        # check
        if not isinstance(filename, str):
            raise TypeError
        if not os.path.isfile(self.dir+filename):
            raise FileNotFoundError(self.dir+filename)
        return Sequence.fromFile(filepath=self.dir+filename, type=self.type)

    def readDataset(self):
        """

        :return:
        """
        files = [self.readFile(filename=filename) for filename in self.getDatasetIterator()]
        return files;

    def applyTransforms(self, output_dir=None):
        # check
        if not output_dir is None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # apply transforms
        sequences = []
        for file in self.readDataset():
            for transform in self.compositeTransform.transforms:
                file.addTransform(transform)
            file.applyTransforms()
            sequences.append(file)
            if not output_dir is None:
                file.save(output_dir=output_dir)
        return sequences

    def plot(self, compared_dataset = None, compared_model = None, sampleName = None, singleMode = True, dimensions=2):
        """

        :return:
        """
        # Check and initilize parameters
        if compared_dataset != None and (not isinstance(compared_dataset, CsvDataset)):
            raise TypeError
        elif compared_dataset != None:
            compared_files = compared_dataset.getDatasetIterator().filenames
        if compared_model != None and not isinstance(compared_model, HiddenMarkovModel):
            raise TypeError

        files = self.readDataset()
        for file in files:
            if sampleName == None or file.filename == sampleName:
                # get plot file
                sequence_plot = file.getPlot(dimensions=dimensions)
                # comparing
                if compared_model != None or compared_dataset != None:
                    # sample generated by model
                    if compared_model != None:
                        compared_plot = CsvDataset.__plot(numpy.array(compared_model.sample()).astype('float'))
                        compared_name = compared_model.name
                    # file from other dataset (choice randomly)
                    if compared_dataset != None:
                        compared_files = compared_dataset.getDatasetIterator()
                        compared_file = compared_dataset.readFile(compared_files[random.randint(0,len(compared_files)-1)])
                        compared_plot = compared_file.getPlot()
                        compared_name = compared_dataset.dir
                    # Plot
                    plt.legend((sequence_plot[0], compared_plot[0]),
                               (self.dir, compared_name), loc='lower right')
                # plot single file
                if singleMode or sampleName != None:
                    plt.title(file.filename)
                    plt.show()
        # Plot all files together
        if not singleMode:
            plt.show()

class Sequence(object):
    """
        Class for file reading and plotting
    """
    def __init__(self, points, filename=None):
        # check
        if not isinstance(points, (numpy.ndarray, list)):
            raise TypeError
        # initialize parameters
        self.points = points
        self.filename = filename
        # transform parameters
        self.compositeTransform = CompositeTransform()

    @classmethod
    def fromFile(cls, filepath=None, type=float):
        """
            take points from file
        :param filepath: file's path
        :param type: data's type (float, int, ecc.)
        :return:
        """
        points,filename=Sequence.__readFile(filepath=filepath, type=type)
        return cls(points, filename)

    # public methods #
    def addTransform(self, transform):
        '''
            add a new transform which will apply to the file
        :param transform:
        :return:
        '''
        self.compositeTransform.addTranform(transform)
    def addTransforms(self, transforms):
        # check
        if not isinstance(transforms, list):
            raise TypeError
        for transform in transforms:
            self.compositeTransform.addTranform(transform)
    def applyTransforms(self, output_dir=None):
        '''
            apply sequentially the specified transforms
        :return:
        '''
        self.points = self.compositeTransform.transform(self.points)
        if output_dir != None:
            self.save(output_dir=output_dir)
    def save(self, output_dir):
        '''
            save the save into the specified directory
        :param output_dir:
        :return:
        '''
        # check if output_dir exists
        if not os.path.exists(output_dir):
            # create dir
            os.makedirs(output_dir)
        #numpy.savetxt(output_dir + self.filename, self.points, delimiter=',')
        with open(output_dir + self.filename, "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(self.points)

    def plot(self, dimensions=2):
        """

        :param dimensions:
        :return:
        """
        # Create and show plot
        self.getPlot()
        plt.show()

    def getPlot(self, dimensions=2):
        """
            provide to plot the list of points in 2 or 3 dimensions
            (of course, the latter depends from the array's dimensions)
        :return:
        """
        switcher = {
            2: Sequence.__plot2D, # 2D
            3: Sequence.__plot3D, # 3D
        }
        # get the function from switcher dictionary
        func = switcher.get(dimensions)
        # Execute the function
        return func(self)

    def getPoints(self, columns=[0,1]):
        """
            returns the points of trajectory. In addition, it allows the devoloper to specify which columns
            she would like to receive.
        :param columns: specifies the selected columns
        :return: a set of points which describe a trajectory
        """
        # check
        if not isinstance(columns,list):
            raise TypeError("dimensions must be a list of integer!")
        # return the sliced array
        # numbers #
        points = []
        for point in self.points:
            points.append(point[columns])
        return numpy.array(points)

    def getIndexPrimitives(self, col=-1):
        """
            return the list of indexies which define the end of a primitive
        :param col: column where to find on which primitive a point belongs
        :return:
        """
        # check
        if not isinstance(col, int):
            print(col)
            raise TypeError
        # get indexes
        index_primitives = self.points[:, col]
        indexes = [x[0] for x,y in zip(enumerate(index_primitives),enumerate(index_primitives[1:])) if x[1]!=y[1]]
        indexes.append(len(self.points))
        # return indexes
        return indexes

    # private methods #
    def __plot2D(self):
        """
            plotting in 2 dimensions
        :return: plot
        """
        plt.scatter(self.points[:, 0], self.points[:, 1])
        for i in range(0, len(self.points)):
            plt.annotate(str(i), (self.points[i, 0], self.points[i, 1]))
        return plt.plot(self.points[:, 0], self.points[:, 1], label=self.filename)
    def __plot3D(self):
        """
            plotting in 3 dimensions
        :return: plot
        """
        plt.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2])
        for i in range(0, len(self.points)):
            plt.annotate(str(i), (self.points[i, 0], self.points[i, 1], self.points[i, 2]))
        return plt.plot(self.points[:, 0], self.points[:, 1], self.points[:, 2], label=self.filename)

    # static methods #
    @staticmethod
    def __readFile(filepath=None, type=float):
        """
            given a the path of a file, provide to read it and return its content
        :param filepath:
        :param type:
        :return:
        """
        # check
        if not isinstance(filepath, str):
            raise TypeError
        if not os.path.isfile(filepath):
            raise FileNotFoundError
        # read file #
        with open(filepath, "r") as file:
            reader = csv.reader(file, delimiter=',')#';')
            vals = list(reader)
            points=numpy.array(vals).astype(type)
        # get filename
        filename = filepath.split('/')[-1]
        return points, filename












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