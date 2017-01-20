from .csv import *
import csv
import numpy
import scipy
import matplotlib.pyplot as plt
import random
import os
import re
from math import sin, cos, radians
from shutil import copyfile
from enum import Enum

random.seed(0)

class Operator(Enum):
    sequence = 1
    iterative = 2
    choice = 3
    disabling = 4
    parallel = 5

class LeapDataset:

    def __init__(self, dir):
        self.dir = dir

    def getCsvDataset(self):
        return CsvDataset(self.dir)

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

    def trasl(self, output_dir, name, dimensions = 1):
        index_file = 0
        for filename in self.getCsvDataset():
            index_file = index_file + 1
            with open(self.dir + filename, "r") as f:
                reader = csv.reader(f, delimiter=',')
                vals = list(reader)
                result = numpy.array(vals).astype('float')

                #for index in range(0, len(result)):
                    #if (dimensions == 1):
                        #result[index][0] = 1
                    #elif (dimensions == 2):
                        #result[index][1] = 1
                    #else:
                        #result[index][2] = 1

                numpy.savetxt(output_dir + name + '_{}.csv'.format(index_file), result, delimiter=',')

    def swap(self, output_dir, name, dimensions = 2):
        # Lettura file
        index_file = 0
        for filename in self.getCsvDataset():
            index_file = index_file + 1
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
                numpy.savetxt(output_dir + name + '_{}.csv'.format(index_file), result, delimiter=',')

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

                x_max = result[maxs[0], 0]
                y_max = result[maxs[1], 1]
                z_max = result[maxs[2], 2]

                x_min = result[mins[0], 0]
                y_min = result[mins[1], 1]
                z_min = result[mins[2], 2]

                if norm_axis:
                    den_x = x_max - x_min
                    den_y = y_max - y_min
                    den_z = z_max - z_min

                    result[:, 0] = (result[:, 0] - x_min) / den_x
                    result[:, 1] = (result[:, 1] - y_min) / den_y
                    result[:, 2] = (result[:, 2] - z_min) / den_z

                    numpy.savetxt(output_dir + filename, result, delimiter=',')

                else :
                    den = max(x_max - x_min, y_max - y_min, z_max - z_min);

                    result[:, 0] = (result[:, 0] - x_min) / den
                    result[:, 1] = (result[:, 1] - y_min) / den
                    result[:, 2] = (result[:, 2] - z_min) / den

                    numpy.savetxt(output_dir + filename, result, delimiter=',')

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

    ### Creazione sequenze ###
    # Sequence
    def sequence_merge(self,  list_dataset, dimensions = 2, scale = 1):
        first_seq = self.read_dataset(dimensions, scale)

        sequences = []
        # Per ogni file in first_seq
        for i in range(0, len(first_seq)):
            seq = first_seq[i]
            # Per ogni elemento presente nella lista
            for file in list_dataset :
                element = file.read_dataset(dimensions, scale)
                random.seed()
                num_rand = int(random.uniform(0, len(element)-1))
                seq = numpy.concatenate((seq, element[num_rand]), axis=0)
            # Salva nuova sequenza
            sequences.append(seq)

        return sequences

    # Iterative
    def iterative_merge(self, iterazioni = 2, dimensions=2, scale=1):
        first_seq = self.read_dataset(dimensions, scale)

        sequences = []
        # Per ogni file
        for i in range(0, len(first_seq)):
            seq = first_seq[i]
            # Genera un numero casuale che indica il file con cui creare l'iterative
            for j in range(0, iterazioni):
                random.seed()
                num_rand = int(random.uniform(0, len(first_seq)-1))
                seq = numpy.concatenate((seq, first_seq[num_rand]), axis = 0)
            sequences.append(seq)

        return sequences

    # Disabling
    def disabling_merge(self, second, third, dimensions=2, scale=1):
        first_seq = self.read_dataset(dimensions, scale)
        second_seq = second.read_dataset(dimensions, scale)
        third_seq = third.read_dataset(dimensions, scale)

        sequences = []

        for i in range(0, len(first_seq)):
            random.seed()
            j = int(random.uniform(0, len(second_seq)-1))
            k = int(random.uniform(0, len(third_seq)-1))
            sequences.append(numpy.concatenate((first_seq[i], second_seq[j], third_seq[k]), axis=0))

        return sequences

    # Parallel
    def parallel_merge(self, second, dimensions=2, scale=1, flag_trasl=False):
        first_seq = self.read_dataset(dimensions, scale)
        second_seq = second.read_dataset(dimensions, scale)

        sequences = []


        for s_first in first_seq:
            num_rand = 0
            if flag_trasl:
                random.seed()
                num_rand = int(random.uniform(1, len(first_seq[0]) / 3))

            s_second = second_seq[int(random.uniform(0, len(second_seq)-1))]
            merge = []

            if len(s_first) <= len(second_seq):
                # Parte iniziale
                for i in range(0, num_rand):

                    merge.append(numpy.concatenate((s_first[i],  s_second[0]), axis=0))
                # Parte centrale
                for i in range(num_rand, len(s_first)):
                    merge.append(numpy.concatenate((s_first[i], s_second[i - num_rand]), axis=0))

                # Parte finale
                for i in range(len(s_first), len(second_seq)):
                    merge.append(numpy.concatenate((s_first[len(s_first)-1], s_second[i]), axis=0))
            else:
                index = 0
                # Parte iniziale
                for i in range(0, num_rand):
                    merge.append(numpy.concatenate((s_first[i], s_second[0]), axis=0))

                # Parte centrale
                for i in range(num_rand, len(s_first)):
                    if i >= len(s_second):
                        index = len(s_second)-1
                    else:
                        index = i

                    merge.append(numpy.concatenate((s_first[i], s_second[index - num_rand]), axis=0))

                # Parte finale
                for i in range(len(second_seq) - num_rand, len(second_seq)):
                    merge.append(numpy.concatenate((s_first[len(s_first) - 1], s_second[i]), axis=0))

            sequences.append(merge)

        return sequences

    # Gesture Folder
    @staticmethod
    def find_gesture_file(path, baseDir, name):
        # Cartella per la gesture
        if not os.path.exists(baseDir + 'original/' + name):
            os.makedirs(baseDir + 'original/' + name)

        index = 0
        # Per ogni sottodirectory
        folders = LeapDataset.get_immediate_subdirectories(path)
        folders = sorted(folders, key=lambda x: (int(re.sub('\D', '', x)), x))# Riordina cartelle
        for folder in folders:
            # Per ogni file
            for file in os.listdir(path+folder):
                if (name+'.csv') == file:
                    index = index + 1
                    copyfile(path+folder+'/'+file, baseDir+'original/'+name+'/'+name+'{}.csv'.format(index))

        return
    # Prende tutte le subdirectories di una data directory
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