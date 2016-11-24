from .csv import *
import csv
import numpy
import scipy
import matplotlib.pyplot as plt
import random
import os
from math import sin, cos, radians
from shutil import copyfile

class LeapDataset:

    def __init__(self, dir):
        self.dir = dir

    def getCsvDataset(self):
        return CsvDataset(self.dir)

    def swap(self, output_dir, name):
        # Lettura file
        index_file = 0
        for filename in self.getCsvDataset():
            index_file = index_file + 1
            with open(self.dir + filename, "r") as f:
                reader = csv.reader(f, delimiter=',')
                vals = list(reader)
                result = numpy.array(vals).astype('float')
                # Swap x con y
                for index in range(0, len(result)):
                    temp = result[index][0]
                    result[index][0] = result[index][1]
                    result[index][1] = temp
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
                den_x = (x_max - x_min)/2
                den_y = (y_max - y_min)/2
                den_z = (z_max - z_min)/2
                # Per ogni punto presente
                theta = radians(degree)
                cosang, sinang = cos(theta), sin(theta)
                for index in range(0, len(result)):
                    matrix_translantion = numpy.asmatrix(numpy.array(
                        [[1, 0, den_x],
                        [0, 1, den_y],
                        [0, 0, 1]]))
                    matrix_translantion_back = numpy.asmatrix(numpy.array(
                        [[1, 0, -den_x],
                        [0, 1, -den_y],
                        [0, 0, 1]]))

                    matrix_rotate = numpy.asmatrix(numpy.array(
                        [[cosang, -sinang, 0],
                        [sinang, cosang, 0],
                        [0,0,1]]))

                    #m = matrix_translantion * matrix_rotate * matrix_translantion_back;
                    m = matrix_rotate
                    result_temp = numpy.array([[0,0,1]])
                    result_temp[0][0]= result[index][0]
                    result_temp[0][1]= result[index][1]
                    t =  result_temp[0]*m
                    result[index][0] = t[0,0]
                    result[index][1] = t[0,1]

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

    # Creazione sequenze
    def sequence_merge(self,  second, dimensions = 2, scale = 1):
        first_seq = self.read_dataset(dimensions, scale)
        second_seq = second.read_dataset(dimensions, scale)

        sequences = []

        for i in range(0, len(first_seq)):
            for j in range(0, len(second_seq)):
                sequences.append(numpy.concatenate((first_seq[i],  second_seq[j]), axis= 0))

        return sequences

    def disabling_merge(self, second, third, dimensions=2, scale=1):
        first_seq = self.read_dataset(dimensions, scale)
        second_seq = second.read_dataset(dimensions, scale)
        third_seq = third.read_dataset(dimensions, scale)

        sequences = []

        for i in range(0, len(first_seq)):
            for j in range(0, len(second_seq)):
                for k in range(0, len(third_seq)):
                    sequences.append(numpy.concatenate((first_seq[i], second_seq[j]), third_seq[k], axis=0))

        return sequences

    def parallel_merge(self, second, dimensions=2, scale=1, flag=False):
        first_seq = self.read_dataset(dimensions, scale)
        second_seq = second.read_dataset(dimensions, scale)

        sequences = []

        if flag:
            for i in range(0, len(first_seq)):
                for j in range(0, len(second_seq)):
                    merge = numpy.c_[first_seq[i], second_seq[j]]
                    sequences.append(merge)
        else:
            random.seed()
            for s_first in first_seq:
                for s_second in second_seq:
                    num_rand = int(random.uniform(1, len(s_first) / 3))

                    merge = []
                    # Parte iniziale
                    for i in range(0, num_rand):
                        merge.append(numpy.concatenate((s_first[i],  s_second[0]), axis=0))
                    # Parte centrale
                    for i in range(num_rand, len(s_first)):
                        merge.append(numpy.concatenate((s_first[i], s_second[i - num_rand]), axis=0))
                    # Parte finale
                    for i in range(len(second_seq) - num_rand, len(second_seq)):
                        merge.append(numpy.concatenate((s_first[len(s_first)-1], s_second[i]), axis=0))

                    sequences.append(merge)

        return sequences


    # Gesture Folder
    def find_gesture_file(self, path, baseDir, name):
        # Cartella per la gesture
        if not os.path.exists(baseDir + 'original/' + name):
            os.makedirs(baseDir + 'original/' + name)

        index = 0
        # Per ogni sottodirectory
        for folder in LeapDataset.get_immediate_subdirectories(self, path):
            # Per ogni file
            for file in os.listdir(path+folder):
                if name in file:
                    index = index + 1
                    copyfile(path+folder+'/'+file, baseDir+'original/'+name+'/'+name+'{}.csv'.format(index))

        return
    # Prende tutte le subdirectories di una data directory
    def get_immediate_subdirectories(self, a_dir):
        return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

    # Replace Csv
    def replace_csv(self, baseDir):
        for file in os.listdir(baseDir):
            with open('/'+baseDir+file) as fin, open('/'+baseDir+'fixed_'+file, 'w') as fout:
                o = csv.writer(fout)
                for line in fin:
                    o.writerow(line.split())
                os.remove('/'+baseDir+file)
                os.rename('/'+baseDir+'fixed_'+file, '/'+baseDir+file)