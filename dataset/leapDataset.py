from .datasetIterator import *
import csv
import numpy
import scipy
import matplotlib.pyplot as plt
import random
import os
import math
import re
from math import sin, cos, radians
from shutil import copyfile
from enum import Enum
import numpy as np

x = np.ones((1000, 1000)) * np.nan


random.seed(0)

class Operator(Enum):
    sequence = 1
    iterative = 2
    choice = 3
    disabling = 4
    parallel = 5

class LeapDataset_2:

    def __init__(self, dir):
        self.dir = dir

    def getCsvDataset(self):
        return DatasetIterator(self.dir)

    def down_sample(self, output_dir, samples):
        for filename in self.getCsvDataset():
            with open(self.dir + filename, "r") as f:
                reader = csv.reader(f, delimiter=',')
                vals = list(reader)
                result = numpy.array(vals).astype('float')
                R = int(1.0 * result[:, 0].size / samples)
                a = numpy.zeros((samples - 1, int(result[0, :].size)))

                for i in range(0, samples - 1):
                    start = i * R
                    end = ((i + 1) * R)
                    a[i, 0] = scipy.nanmean(result[start:end, 0])
                    a[i, 1] = scipy.nanmean(result[start:end, 1])
                    a[i, 2] = scipy.nanmean(result[start:end, 2])

                numpy.savetxt(output_dir + filename, a, delimiter=',')

    def swap(self, output_dir, name, dimensions = 2):
        # Lettura file
        for filename in self.getCsvDataset():
            items = re.findall('\d*\D+', filename)# Nome file
            # Lettura file
            with open(self.dir + filename, "r") as f:
                reader = csv.reader(f, delimiter=',')
                vals = list(reader)
                result = numpy.array(vals).astype('float')

                if dimensions == 2:
                    # Swap x con y
                    for index in range(0, len(result)):
                        temp = result[index][0]
                        result[index][0] = result[index][1]
                        result[index][1] = temp
                elif dimensions == 3:
                    # Swap x con z
                    for index in range(0, len(result)):
                        temp = result[index][0]
                        result[index][0] = result[index][2]
                        result[index][2] = temp

                # Salva
                #numpy.savetxt(output_dir + name + '_{}.csv'.format(index_file), result, delimiter=',')
                numpy.savetxt(output_dir + name + '_' + items[len(items)-1], result, delimiter=',')

    def rotate_lines(self, output_dir, name, degree = 0):
        # Lettura file
        index_file = 0
        for filename in self.getCsvDataset():
            index_file = index_file + 1
            with open(self.dir + filename, "r") as f:
                reader = csv.reader(f, delimiter=',')
                vals = list(reader)
                result = numpy.array(vals).astype('float')
                # Massimi e minimi
                maxs = result.argmax(axis=0);
                mins = result.argmin(axis=0);
                x_max = result[maxs[0], 0]
                y_max = result[maxs[1], 1]
                z_max = result[maxs[2], 2]
                x_min = result[mins[0], 0]
                y_min = result[mins[1], 1]
                z_min = result[mins[2], 2]
                den_x = (x_max + x_min)/2
                den_y = (y_max + y_min)/2
                den_z = (z_max + z_min)/2

                theta = radians(degree)
                cosang, sinang = cos(theta), sin(theta)
                matrix_translantion = numpy.asmatrix(numpy.array(
                    [[1, 0, den_x],
                     [0, 1, den_y],
                     [0, 0, 1]]))
                matrix_translantion_back = numpy.asmatrix(numpy.array(
                    [[1, 0, -den_x],
                     [0, 1, -den_y],
                     [0, 0, 1]]))

                matrix_rotate = numpy.asmatrix(numpy.array(
                    [[cosang, - sinang, 0],
                     [sinang, cosang, 0],
                     [0, 0, 1]]))

                m = matrix_translantion * matrix_rotate * matrix_translantion_back;

                # Per ogni punto presente
                for index in range(0, len(result)):

                    result_temp = numpy.array([[0,0,1]])
                    result_temp[0][0]= result[index][0]
                    result_temp[0][1]= result[index][1]
                    t =  m * numpy.matrix(result_temp[0]).T
                    result[index][0] = t[0]
                    result[index][1] = t[1]

                # Salva
                numpy.savetxt(output_dir + name +'_{}.csv'.format(index_file), result, delimiter=',')

    def normalise(self, output_dir, norm_axis = False):
        for filename in self.getCsvDataset():
            with open(self.dir + filename, "r") as f:
                reader = csv.reader(f, delimiter=',')
                vals = list(reader)
                result = numpy.array(vals).astype('float')

                maxs = result.argmax(axis=0);
                mins = result.argmin(axis=0);
                # Max
                x_max = result[maxs[0], 0]
                y_max = result[maxs[1], 1]
                z_max = result[maxs[2], 2]
                # Min
                x_min = result[mins[0], 0]
                y_min = result[mins[1], 1]
                z_min = result[mins[2], 2]

                # X Y Z
                if norm_axis:
                    den_x = x_max - x_min
                    den_y = y_max - y_min
                    den_z = z_max - z_min

                    result[:,0] = (result[:, 0] - x_min) / den_x
                    result[:,1] = (result[:, 1] - y_min) / den_y
                    result[:,2] = (result[:, 2] - z_min) / den_z

                    numpy.savetxt(output_dir + filename, result, delimiter=',')

                else:
                    den = max(x_max - x_min, y_max - y_min, z_max - z_min);

                    result[:, 0] = (result[:, 0] - x_min) / den
                    result[:, 1] = (result[:, 1] - y_min) / den
                    result[:, 2] = (result[:, 2] - z_min) / den

                    # delta
                    #for i in range(1, result[:, 0].size):
                    #    result[i, 0] = result[i, 0] - result[i-1, 0]
                    #    result[i, 1] = result[i, 1] - result[i-1, 1]

                    # Time
                    #result2 = result[:, 3]
                    #for i in range(2, len(result2)):
                    #numbers = []
                    #for j in range(1, i):
                    #numbers.append(result2[j])
                    #mean = float(sum(numbers)) / max(len(numbers), 1)
                    #dim = 1/(i-1)
                    #s = 0
                    #for j in range(0, i):
                    #d = result2[i]-mean
                    #p = math.pow(d, 2)
                    #s = s + p
                    #result_[i, 3] = math.sqrt(dim * s)

                    numpy.savetxt(output_dir + filename, result, delimiter=',')

    def save_file_csv(self, path):
        # Save csv file
        return

    def read_dataset(self, dimensions = 2, scale = 1):
        sequences = [];
        for filename in self.getCsvDataset():
            seq = self.read_file(filename, dimensions, scale);
            sequences.append(seq)
        return sequences;

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

    def leave_one_out(self, leave_index = 0, dimensions = 2, scale = 1):
        sequences = self.read_dataset(dimensions=dimensions, scale=scale)
        leave = sequences.pop(leave_index)
        return leave, sequences

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



    # Gesture Folder
    @staticmethod
    def find_gesture_file(path, baseDir, name):
        # Create Original Folder
        if not os.path.exists(baseDir + 'original/' + name):
            os.makedirs(baseDir + 'original/' + name)

        # Copy files from original dataset to original folder
        index = 0
        # For each folders
        folders = LeapDataset.get_immediate_subdirectories(path)
        folders = sorted(folders, key=lambda x: (str(re.sub('\D', '', x)), x))# Riordina cartelle
        for folder in folders:
            for file in os.listdir(path+folder):
                if (name+'.csv') == file:
                    index = index + 1
                    #copyfile(path+folder+'/'+file, baseDir+'original/'+name+'/'+name+'{}.csv'.format(index))
                    copyfile(path+folder+'/'+file, baseDir+'original/'+name+'/'+name+'_' + folder+'.csv')

        return

    # Get all subdirectories from the choosen directory
    @staticmethod
    def get_immediate_subdirectories(baseDir):
        return [name for name in os.listdir(baseDir)
            if os.path.isdir(os.path.join(baseDir, name))]

    # Replace Csv
    @staticmethod
    def replace_csv(baseDir):
        for file in os.listdir(baseDir):
            with open('/'+baseDir+file) as fin, open('/'+baseDir+'fixed_'+file, 'w') as fout:
                o = csv.writer(fout)
                for line in fin:
                    o.writerow(line.split())
                os.remove('/'+baseDir+file)
                os.rename('/'+baseDir+'fixed_'+file, '/'+baseDir+file)

    @staticmethod
    def gen_random_name(nome, gestures):
        num_rand_1 = int(random.uniform(0, len(nome) - 1))
        index = -1

        while LeapDataset.check(num_rand_1, nome, gestures) :
            index = index + 1
            num_rand_1 = index

        return num_rand_1

    @staticmethod
    def check(num_rand_1, nome, gestures):
        for i in range(0, len(gestures)):
            if nome[num_rand_1] in gestures[i]:
                return True
        return False

    # Create the folder for origina, normalize and down-trajectory
    @staticmethod
    def create_folder(baseDir, nome, operator = '', is_ground = True):
        if(is_ground == True):
            # Creazione cartelle
            if not os.path.exists(baseDir + 'original/' + nome):
                os.makedirs(baseDir + 'original/' + nome)
            if not os.path.exists(baseDir + 'normalised-trajectory/' + nome):
                os.makedirs(baseDir + 'normalised-trajectory/' + nome)
            if not os.path.exists(baseDir + 'down-trajectory/' + nome):
                os.makedirs(baseDir + 'down-trajectory/' + nome)
        else:
            # Creazione cartelle per gesture composte
            if not os.path.exists(baseDir + operator+'/'+nome):
                os.makedirs(baseDir + operator+'/'+nome)

    # Plot dataset
    @staticmethod
    def plot_samples(baseDir, path):
        dataset = LeapDataset(baseDir + 'down-trajectory/' + path + '/')
        lenght = len(dataset.getDatasetIterator().filenames)
        delta = 1#(int)(lenght/3)

        for i in range(1, lenght, delta):
            sequence = dataset.read_file(dataset.getDatasetIterator().filenames[i], dimensions=2, scale=100)
            plt.axis("equal")
            plt.plot(sequence[:, 0], sequence[:, 1])
            plt.show()