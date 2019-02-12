from dataset import *
import random

### Create synthetic datasets  ###

random.seed()

# Choice
class MergeChoiceDataset:
    # Composes the sequence of two or more gesture to make a choice sequence.
    @staticmethod
    def create_merge_dataset(list_dataset, filepath, cols=[0,1]):
        if not isinstance(list_dataset, list):
            raise TypeError
        if not isinstance(cols, list):
            raise TypeError

        # Gets sequences
        sequences = []
        for dataset in list_dataset:
            dataset_sequences = []
            for sequence in dataset.readDataset():
                dataset_sequences.append(sequence)
            sequences.append(dataset_sequences)

        # Creates dataset
        new_sequences = []
        for i in range(0, len(sequences[0])):
            # Choices a gesture from the list_dataset
            gesture_index = int(random.uniform(0, len(list_dataset)))
            # and choices a file of the selected gesture
            file_index = int(random.uniform(0, len(sequences[gesture_index])))
            # Add sequence
            new_sequences.append(sequences[gesture_index][file_index])

        # Save file
        for i in range(0, len(new_sequences)):
            numpy.savetxt(filepath+'_{}.csv'.format(i), new_sequences[i], delimiter=',')

# Disabling
class MergeDisablingDataset:
    # Composes the sequence of all files to make a disabling sequence
    @staticmethod
    def create_disabling_dataset(list_dataset, filepath, cols=[0,1]):
        if not isinstance(list_dataset, list):
            raise TypeError
        if not isinstance(cols, list):
            raise TypeError

        first_sequences = list_dataset[0].readDataset()
        second_sequences = list_dataset[1].readDataset()
        sequences = []
        for i in range(0, len(first_sequences)):
            j = int(random.uniform(0, len(second_sequences)-1))
            sequences.append(numpy.concatenate((first_sequences[i], second_sequences[j]), axis=0))

        # Save file
        for i in range(0, len(sequences)):
            numpy.savetxt(filepath+'_{}.csv'.format(i), sequences[i], delimiter=',')

# Iterative
class MergeIterativeDataset:
    # Provides to compose the sequence of all files in a single sequence. self's files start the sequence.
    @staticmethod
    def create_iterative_dataset(list_dataset, filepath, cols=[0,1], iterations = 1):
        if not isinstance(cols, list):
            raise TypeError

        # Composes an iteration of each file with a random file (of same dataset)
        dataset_seq = list_dataset.readDataset()
        length = len(dataset_seq)
        sequences = []
        for sequence in list_dataset.readDataset():
            new_sequence = deepcopy(sequence)
            # Select a different random file for each iteration
            for j in range(0, iterations):
                num_rand = int(random.uniform(0, length-1))
                # Concatenate
                new_sequence = numpy.concatenate((new_sequence, dataset_seq[num_rand]), axis = 0)
            sequences.append(new_sequence)

        # Save file
        for i in range(0, len(sequences)):
            numpy.savetxt(filepath+'_{}.csv'.format(i), sequences[i], delimiter=',')

# Parallel
class MergeParallelDataset:
    @staticmethod
    def create_parallel_dataset(list_dataset, filepath, cols=[0,1], flag_trasl=False, type="unistroke"):
        original_seq = []
        for index in range(0, len(list_dataset)):
            original_seq.append(list_dataset[index].readDataset())

        sequences = []
        for s_first in original_seq[0]:
            num_rand = 0
            # Add translation if flag is true
            if flag_trasl:
                num_rand = int(random.uniform(1, len(first_seq[0]) / 3))
            # Array for the new sequence
            merge = deepcopy(s_first)

            for index in range(1, len(list_dataset)):
                s_second = original_seq[index][int(random.uniform(0, len(original_seq[index])))]
                merge_temp = []

                # We can have two different situations:
                # i) the first sequence is shorter than the second sequence.
                # ii) viceversa.
                # In each case we need two sequences with the same lenght
                if len(s_first) <= len(s_second):
                    # Starting part
                    for i in range(0, num_rand):
                        merge_temp.append(numpy.concatenate((merge[i],  s_second[0]), axis=0))
                    # Central part
                    for i in range(num_rand, len(s_first)):
                        merge_temp.append(numpy.concatenate((merge[i], s_second[i - num_rand]), axis=0))
                    # Finally part
                    for i in range(len(s_first), len(s_second)):
                        merge_temp.append(numpy.concatenate((merge[len(s_first)-1], s_second[i]), axis=0))
                else:
                    index = 0
                    # Starting part
                    for i in range(0, num_rand):
                        merge_temp.append(numpy.concatenate((merge[i], s_second[0]), axis=0))
                    # Central part
                    for i in range(num_rand, len(s_first)):
                        if i >= len(s_second):
                            j = len(s_second)-1
                        else:
                            j = i
                        merge_temp.append(numpy.concatenate((merge[i], s_second[j - num_rand]), axis=0))
                    # Finally part
                    for i in range(len(s_second) - num_rand, len(s_second)):
                        merge_temp.append(numpy.concatenate((merge[len(s_first) - 1], s_second[i]), axis=0))

                # If we are working on multistroke dataset, this function changes the number of strokes:
                # for each frame the new stroke number is obtained by adding the last stroke number of new_sequence.
                if type != "unistroke":
                    for index in range(0, len(merge_temp)):
                        merge_temp[index][-1] = merge_temp[index][-1] + merge_temp[-1][2]
                merge = merge_temp

            sequences.append(merge)

        # Save file
        for i in range(0, len(sequences)):
            numpy.savetxt(filepath+'_{}.csv'.format(i), sequences[i], delimiter=',')


# Sequence
class MergeSequenceDataset:
    # Provides to compose the sequence of all files in a single sequence. self's files start the sequence.
    @staticmethod
    def create_sequence_dataset(list_dataset, filepath, cols=[0,1], type="unistroke"):
        if not isinstance(list_dataset, list):
            raise TypeError
        if not isinstance(cols, list):
            raise TypeError

        # Read dataset
        dataset = []
        for data in list_dataset:
            dataset.append(data.readDataset())

        # Composes first dataset's files with each one of the other dataset
        sequences = []
        for sequence in dataset[0]:
            new_sequence = deepcopy(sequence)
            for i in range(1, len(dataset)):
                # Gets a random sequence from the dataset
                num_rand = int(random.uniform(0, len(dataset[i])-1))
                sequence_to_concatenate = deepcopy(dataset[i][num_rand])

                # If we are working on multistroke dataset, this function changes the number of strokes:
                # for each frame the new stroke number is obtained by adding the last stroke number of new_sequence.
                if type != "unistroke":
                    for index in range(0, len(sequence_to_concatenate)):
                        sequence_to_concatenate[index][-1] = sequence_to_concatenate[index][-1] + new_sequence[-1][-1]

                new_sequence = numpy.concatenate((new_sequence, sequence_to_concatenate), axis=0)
            # Add new sequence
            sequences.append(new_sequence)

        # Save file
        for i in range(0, len(sequences)):
            numpy.savetxt(filepath+'_{}.csv'.format(i), sequences[i], delimiter=',')