from .csv import *
import csv
import numpy
import scipy
import matplotlib.pyplot as plt
import random

class LeapDataset:

    def __init__(self, dir):
        self.dir = dir

    def getCsvDataset(self):
        return CsvDataset(self.dir)

    def normalise(self, output_dir):
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
                z_min = result[maxs[2], 2]

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

    def read_file_parallel(self, filename, dimensions=4, scale=1):
        with open(self.dir + filename, "r") as f:
            reader = csv.reader(f, delimiter=',')
            vals = list(reader)

            result = numpy.array(vals).astype('float')
            merge = []
            for item in result:
                seq_1 = [item[0] * scale, item[1] * scale]
                seq_2 = [item[2] * scale, item[3] * scale]
                merge_row = numpy.array([seq_1, seq_2])

                merge.append(merge_row)

            return merge

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

    def parallel_merge(self, second, dimensions=2, scale=1, dis=False):
        first_seq = self.read_dataset(dimensions, scale)
        second_seq = second.read_dataset(dimensions, scale)

        sequences = []

        if dis:
            for i in range(0, len(first_seq)):
                for j in range(0, len(second_seq)):
                    merge = numpy.c_[first_seq[i], second_seq[j]]
                    sequences.append(merge)
        else:
            for seq_first in first_seq:
                for seq_second in second_seq:
                    num_rand = int(random.uniform(1, len(seq_first) / 3))

                    # Parte iniziale
                    for i in range(0, num_rand):
                        merge = numpy.append(numpy.c_[seq_first[i], seq_second[0]])
                    # Parte centrale
                    for i in range(num_rand, len(seq_first)):
                        merge = numpy.append(numpy.c_[seq_first[i], seq_second[i - num_rand]])
                    # Parte finale
                    for i in range(len(second_seq) - num_rand, len(second_seq)):
                        merge = numpy.append(numpy.c_[seq_first[len(seq_first - 1)], seq_second[i]])

                    sequences.append(merge)

        return sequences



