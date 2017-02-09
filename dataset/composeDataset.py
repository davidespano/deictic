
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

class composeDataset:

    def __init__(self, dir):
        self.dir = dir

    # Return all directory's csv files
    def getCsvDataset(self):
        return DatasetIterator(self.dir)

######### Composing Files #########
    ## Sequence
    # Provides to compose the sequence of all files in a single sequence. self's files start the sequence.
    def sequence_merge(self,  list_dataset, baseDir, dimensions = 2, scale = 1):
        first_seq = self.read_dataset(dimensions, scale)# Get all first dataset's files

        # List of new sequences
        sequences = []
        # Filename
        file = self[0]
        items = re.findall('\d*\D+', file)# filename
        name = items[len(items)-1]

        # Composes first dataset's files with each one of the other dataset
        for i in range(0, len(first_seq)):
            seq = first_seq[i]

            for dataset in (baseDir+'/'+list_dataset):
                name = name+'_'+dataset
                for file in (baseDir+'/'+list_dataset+'/'+dataset):
                    element = file.read_dataset(dimensions, scale)
                    random.seed()
                    num_rand = int(random.uniform(0, len(element)-1))
                    seq = numpy.concatenate((seq, element[num_rand]), axis=0)

            # Add sequence in list
            sequences.append(seq)

        # Save and return sequence
        for i in range(0, len(sequences)):
            numpy.savetxt(baseDir + '_sequence_' + name + '_{}.csv'.format(i), sequences[i], delimiter=',')
        return sequences

    ## Iterative
    # Provides to compose a iteratively single file.
    def iterative(self, baseDir, iterations = 2, dimensions=2, scale=1):
        first_seq = self.read_dataset(dimensions, scale)

        # List of new sequences
        sequences = []
        # Filename
        file = self[0]
        items = re.findall('\d*\D+', file)# filename
        name = items[len(items)-1]

        # Composes an iteration of each file with a random file (of same dataset)
        for i in range(0, len(first_seq)):
            seq = first_seq[i]
            # Select a different random file for each iteration
            for j in range(0, iterations):
                random.seed()
                num_rand = int(random.uniform(0, len(first_seq)-1))
                # Concatenate
                seq = numpy.concatenate((seq, first_seq[num_rand]), axis = 0)
            sequences.append(seq)

        # Save and return sequence
        for i in range(0, len(sequences)):
            numpy.savetxt(baseDir + '_iterative_' + name + '_{}.csv'.format(i), sequences[i], delimiter=',')
        return sequences

    ## Disabling
    # Composes three files (from self, second and third dataset) to make a disabling sequence
    def disabling_merge(self, second, third, baseDir, dimensions=2, scale=1):
        first_seq = self.read_dataset(dimensions, scale)
        second_seq = second.read_dataset(dimensions, scale)
        third_seq = third.read_dataset(dimensions, scale)

        # List of new sequences
        sequences = []
        # Filename
        file = self[0]
        items = re.findall('\d*\D+', file)# self
        name = items[len(items)-1]
        file = second[0]
        items = re.findall('\d*\D+', file)# second
        name = name+'_'+items[len(items)-1]
        file = third[0]
        items = re.findall('\d*\D+', file)# third
        name = name+'_'+items[len(items)-1]

        # Concatenate files (the files from second and third dataset are choosen randomly)
        for i in range(0, len(first_seq)):
            random.seed()
            j = int(random.uniform(0, len(second_seq)-1))
            k = int(random.uniform(0, len(third_seq)-1))
            sequences.append(numpy.concatenate((first_seq[i], second_seq[j], third_seq[k]), axis=0))

        # Save and return sequence
        for i in range(0, len(sequences)):
            numpy.savetxt(baseDir + '_disabling_' + name + '_{}.csv'.format(i), sequences[i], delimiter=',')
        return sequences

    ## Parallel
    # Composes the files of two dataset in a single parallel file.
    def parallel_merge(self, second, baseDir, dimensions=2, scale=1, flag_transl=False):
        first_seq = self.read_dataset(dimensions, scale)
        second_seq = second.read_dataset(dimensions, scale)

        # List of new sequences
        sequences = []
        # Filename
        file = self[0]
        items = re.findall('\d*\D+', file)# self
        name = items[len(items)-1]
        file = second[0]
        items = re.findall('\d*\D+', file)# second
        name = name+'_'+items[len(items)-1]

        # Composes files
        for s_first in first_seq:
            num_rand = 0
            random.seed()

            # Add translation if flag is true
            if flag_transl:
                num_rand = int(random.uniform(1, len(first_seq[0]) / 3))
            # The file from second dataset is choosen randomly
            s_second = second_seq[int(random.uniform(0, len(second_seq)-1))]
            # Array for the new sequence
            merge = []

            # We can have two different situations:
            # i) the first sequence is shortly than the second sequence.
            # ii) viceversa.
            # In each case we need two sequence with the same lenght
            if len(s_first) <= len(second_seq):
                # Starting part
                for i in range(0, num_rand):

                    merge.append(numpy.concatenate((s_first[i],  s_second[0]), axis=0))
                # Central part
                for i in range(num_rand, len(s_first)):
                    merge.append(numpy.concatenate((s_first[i], s_second[i - num_rand]), axis=0))

                # Finally part
                for i in range(len(s_first), len(second_seq)):
                    merge.append(numpy.concatenate((s_first[len(s_first)-1], s_second[i]), axis=0))
            else:
                index = 0
                # Starting part
                for i in range(0, num_rand):
                    merge.append(numpy.concatenate((s_first[i], s_second[0]), axis=0))

                # Central part
                for i in range(num_rand, len(s_first)):
                    if i >= len(s_second):
                        index = len(s_second)-1
                    else:
                        index = i

                    merge.append(numpy.concatenate((s_first[i], s_second[index - num_rand]), axis=0))

                # Finally part
                for i in range(len(second_seq) - num_rand, len(second_seq)):
                    merge.append(numpy.concatenate((s_first[len(s_first) - 1], s_second[i]), axis=0))

            sequences.append(merge)


        # Save and return sequence
        for i in range(0, len(sequences)):
            numpy.savetxt(baseDir + '_parallel_' + name + '_{}.csv'.format(i), sequences[i], delimiter=',')
        return sequences