from .csv import *
import csv
import numpy
import matplotlib.pyplot as plt
import os
import re
from lxml import etree
from shutil import copyfile
from enum import Enum

class TypeFile(Enum):
    csv = 0
    xml = 1

class ToolsDataset:

    def __init__(self, dir):
        self.dir = dir

    # Return all dataset's csv files
    def getCsvDataset(self):
        return CsvDataset(self.dir)

    # Return all dataset's files
    def read_dataset(self, dimensions = 2, scale = 1):
        sequences = [];
        for filename in self.getCsvDataset():
            seq = self.read_file(filename, dimensions, scale);
            sequences.append(seq)
        return sequences;

    # Read file
    def read_file(self, filename, dimensions = 2, scale = 1):
        with open(self.dir + filename, "r") as f:
            reader = csv.reader(f, delimiter=',')
            vals = list(reader)
            result = numpy.array(vals).astype('float')
            if dimensions == 2:
                seq = numpy.column_stack((result[:, 0] * scale, result[:, 1] * scale))
            if dimensions == 3:
                seq = numpy.column_stack((result[:, 0] * scale, result[:, 1] * scale, result[:, 2] * scale))
            if dimensions == 4:
                seq = numpy.column_stack(
                    (result[:, 0] * scale, result[:, 1] * scale, result[:, 2] * scale, result[:, 3] * scale))
            return seq

    # leave_one_out_dataset
    # Returns two dataset elements: the file for testing e and the files for training.
    def leave_one_out_dataset(self, leave_index = 0, dimensions = 2, scale = 1):
        # Gets all files
        sequences = self.read_dataset(dimensions=dimensions, scale=scale)
        # Removes file at the specified index
        leave = sequences.pop(leave_index)

        return leave, sequences

    # Plot
    # Plots input dataset's files
    def plot(self, dimensions = 2):
        fig = plt.figure();
        ax = None
        if dimensions == 3:
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
        else:
            plt.axis('equal')

        for filename in self.getCsvDataset():
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
    @staticmethod
    def plot_samples(baseDir, path):
        dataset = ToolsDataset(baseDir + 'down-trajectory/' + path + '/')
        lenght = len(dataset.getCsvDataset().filenames)
        delta = (int)(lenght/3)

        for i in range(1, lenght, delta):
            sequence = dataset.read_file(dataset.getCsvDataset().filenames[i], dimensions=2, scale=100)
            plt.axis("equal")
            plt.plot(sequence[:, 0], sequence[:, 1])
            plt.show()
    ## plot_gesture
    # Plots the model input examples.
    @staticmethod
    # Plot a model gesture
    def plot_gesture(model):
        for i in range(1, 3):
            sequence = model.sample()
            result = numpy.array(sequence).astype('float')
            plt.axis("equal")
            plt.plot(result[:, 0], result[:, 1])
            plt.show()


    # Find_Gesture_File
    # Is used to get the original dataset files and copy them in a new apposite folder.
    @staticmethod
    def find_gesture_file(path, baseDir, name):
        # Create Original Folder
        if not os.path.exists(baseDir + 'original/' + name):
            os.makedirs(baseDir + 'original/' + name)

        # Copy files from original dataset to original folder
        index = 0
        # For each folders
        folders = ToolsDataset.get_subdirectories(path)
        folders = sorted(folders, key=lambda x: (str(re.sub('\D', '', x)), x))# Riordina cartelle
        for folder in folders:
            for file in os.listdir(path+folder):
                if (name+'.csv') == file:
                    index = index + 1
                    #copyfile(path+folder+'/'+file, baseDir+'original/'+name+'/'+name+'{}.csv'.format(index))
                    copyfile(path+folder+'/'+file, baseDir+'original/'+name+'/'+name+'_' + folder+'.csv')
        return

    # Get_SubDirectories
    # Get all subdirectories from the choosen directory
    @staticmethod
    def get_subdirectories(baseDir):
        return [name for name in os.listdir(baseDir)
                if os.path.isdir(os.path.join(baseDir, name))]

    # Replace CSV
    # Is used to change the format csv files. It is necessary if files don't have commas or spaces.
    @staticmethod
    def replace_csv(baseDir):
        # For each files in the directory
        for file in os.listdir(baseDir):
            # Open and write file
            with open('/'+baseDir+file) as fin, open('/'+baseDir+'fixed_'+file, 'w') as fout:
                o = csv.writer(fout)
                for line in fin:
                    o.writerow(line.split())
                # Remove old file
                os.remove('/'+baseDir+file)
                # Rename file
                os.rename('/'+baseDir+'fixed_'+file, '/'+baseDir+file)

    # Create, if not exist, the folder original, normalize and down-trajectory for the new gesture
    @staticmethod
    def create_folder(baseDir, gesture_name):
        # Folders
        if not os.path.exists(baseDir + 'original/' + gesture_name):
            os.makedirs(baseDir + 'original/' + gesture_name)
        if not os.path.exists(baseDir + 'normalised-trajectory/' + gesture_name):
            os.makedirs(baseDir + 'normalised-trajectory/' + gesture_name)
        if not os.path.exists(baseDir + 'down-trajectory/' + gesture_name):
            os.makedirs(baseDir + 'down-trajectory/' + gesture_name)

    # Xml to CSV
    # Converts input gesture xml files to csv files
    @staticmethod
    def xml_to_csv(datasetDir, gestureName, baseDir):

        for file in datasetDir:
            data = open(datasetDir+'/conversion.xslt')
            xslt_content = data.read()
            xslt_root = etree.XML(xslt_content)
            dom = etree.parse(datasetDir+'/'+gestureName)
            transform = etree.XSLT(xslt_root)
            result = transform(dom)
            f = open(baseDir+'/'+gestureName+'/'+file, 'w')
            f.write(str(result))
            f.close()
        return