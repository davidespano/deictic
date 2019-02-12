# trasnform methods #
from dataset import CsvDatasetExtended, CsvDataset, NormaliseLengthTransform, ScaleDatasetTransform, CenteringTransform, \
    ResampleInSpaceTransformOnline, ResampleInSpaceTransform, ResampleInSpaceTransformMultiStroke
# config #
from config import Config
# imports #
import numpy
import random
import os
import copy
import csv
random.seed()

# todo: tutorial #


# Creates two file, train and test, for each k iteration. The [] file is used for training phase and the latter for testing.
# name          -> the list of gesture.
# inputBase     -> contains the file of each gesture.
# outputBase    -> this folder will contain the test cross validation files.
def ten_cross_validation_factory(names, inputBase, outputBase, k = 10):
    for index in range(0, k):
        for name in names:
            dataset = CsvDataset(inputBase+name)
            # Creates the folder for the index iteration
            if not os.path.exists(outputBase+name):
                os.makedirs(outputBase+name)
            # Creates test and training list
            dataset.ten_cross_validation(outputBase+'/'+name+'/', k=index)


# Creates the synthetic dataset. A synthetic dataset is obtained combining different gestures between each other. For the moment we could not combine
# unistroke gesture with multistroke gesture.
# inputBase     -> contains the file of each gesture.
# outputBase    -> this folder will contain the test cross validation files.
# name          -> the list of gestures.
# iter          -> how many gesture will create.
# type          -> the type of gesture (unistroke or multistroke)
def synthetic_dataset_factory(names, inputBase, outputBase, iter, type='unistroke', operator = [0,1,2,3]):
    # Get all gesture's dataset
    list_dataset = []
    for name in names:
        list_dataset.append(CsvDataset(inputBase+name+'/'))
    indices = []
    for i in range(0, len(list_dataset)):
        indices.append(i)

    for index in range(0, iter):
        # Get a random index for dataset
        tmp_indices = deepcopy(indices)
        num_rand_1 = tmp_indices.pop(int(random.uniform(0, len(tmp_indices)-1)))
        num_rand_2 = tmp_indices.pop(int(random.uniform(0, len(tmp_indices)-1)))

        # Choice
        if(0 in operator):
            list = []
            list.append(list_dataset[num_rand_1])
            list.append(list_dataset[num_rand_2])
            # Filename
            filename = 'choice-' +type+'-'+ names[num_rand_1] +'-'+type+'-'+ names[num_rand_2]
            #filename = ''
            #for i in range(0, int(random.uniform(1, len(list_dataset)))):
            #    num_rand = int(random.uniform(0, len(list_dataset)-1))
            #    filename = 'parallel_'+filename + '_' + names[num_rand]
            #    list.append(list_dataset[num_rand])
            if not os.path.exists(outputBase + filename):
                os.makedirs(outputBase + filename)
            # Makes files
            MergeChoiceDataset.create_merge_dataset(list, outputBase + filename +'/'+ filename)

            print('dataset ' + filename + ' created')

        # Iterative
        if (1 in operator):
            filename = 'iterative-' +type+'-'+ names[num_rand_1]
            if not os.path.exists(outputBase + filename):
                os.makedirs(outputBase + filename)
            # Makes files
            MergeIterativeDataset.create_iterative_dataset(list_dataset[num_rand_1],
                                                           outputBase + filename +'/'+ filename)
            print('dataset ' + filename + ' created')

        # Parallel
        if (2 in operator):
            list = []
            list.append(list_dataset[num_rand_1])
            list.append(list_dataset[num_rand_2])
            # Filename
            filename = 'parallel-' +type+'-'+ names[num_rand_1] +'-'+type+'-'+ names[num_rand_2]
            #filename = ''
            #for i in range(0, int(random.uniform(1, len(list_dataset)))):
            #    num_rand = int(random.uniform(0, len(list_dataset)-1))
            #    filename = 'parallel_'+filename + '_' + names[num_rand]
            #    list.append(list_dataset[num_rand])
            if not os.path.exists(outputBase + filename):
                os.makedirs(outputBase + filename)
            # Makes files
            MergeParallelDataset.create_parallel_dataset(list,
                                                         outputBase + filename +'/'+ filename,
                                                         flag_trasl=False, type = type)
            print('dataset ' + filename + ' created')

        # Sequence
        if (3 in operator):
            list = []
            list.append(list_dataset[num_rand_1])
            list.append(list_dataset[num_rand_2])
            filename = 'sequence-' +type+'-'+ names[num_rand_1] +'-'+type+'-'+ names[num_rand_2]
            #for i in range(0, int(random.uniform(1, len(list_dataset)))):
            #    num_rand = int(random.uniform(0, len(list_dataset)-1))
            #    filename = 'sequence_'+filename + '_' + names[num_rand]
            #    list.append(list_dataset[num_rand])
            if not os.path.exists(outputBase + filename):
                os.makedirs(outputBase + filename)
            #print('making ' + filename + ' dataset')
            # Makes files
            MergeSequenceDataset.create_sequence_dataset(list,
                                                         outputBase + filename +'/'+ filename,
                                                         type = type)
            print('dataset ' + filename + ' created')


# Starting from the raw dataset, the function applies a list of transformations on the selected dataset.
# names         -> the list of files.
# inputBase     -> contains the file of each gesture.
# outputBase    -> this folder will contain the test cross validation files.
# type          -> the type of gesture (unistroke or multistroke).
def dataset_factory(names, inputDir, outputDir, unistroke_mode = True):

    for gesture in names:
        print('Making '+gesture[0]+' dataset')
        input_dir = inputDir+gesture[0]+'/'
        output_dir = outputDir+gesture[0]+'/'
        dataset = CsvDataset(input_dir, type=float)

        # Transform
        transform1 = NormaliseLengthTransform(axisMode=True)
        transform2 = ScaleDatasetTransform(scale=100)
        transform3 = CenteringTransform()
        if unistroke_mode:
            transform5 = ResampleInSpaceTransform(samples=gesture[1])
        else:
            transform5 = ResampleInSpaceTransformMultiStroke(samples=gesture[1], strokes = gesture[2])
        # Apply transforms
        dataset.addTransform(transform1)
        dataset.addTransform(transform2)
        dataset.addTransform(transform3)
        dataset.addTransform(transform5)

        dataset.applyTransforms(output_dir)

        print("Dataset "+gesture[0]+" completed")
    return

mode = 9
n_sample = 20

########################################## Deictic Dataset ##########################################################
# Unica
if mode == 1:
    list_gesture = {("rectangle", 4*n_sample), ("triangle", 3*n_sample), ("caret", 2*n_sample), ("v", 2*n_sample), ("x", 3*n_sample),
                    ("left_sq_bracket", 3*n_sample), ("right_sq_bracket", 3*n_sample), ("delete_mark", 3*n_sample),
                    ("star", 5*n_sample), ('check', 2*n_sample)
                    }#, ("left", n_sample), ("right", n_sample)
    dataset_factory(list_gesture, Config.baseDir+'deictic/unica-dataset/raw/', baseDir+'deictic/unica-dataset/resampled/')

# 1Dollar
if mode == 2:
    # name and n_samples
    list_gesture = {("arrow", 4*n_sample), ("caret", 2*n_sample), ("circle", 4*n_sample), ("check", 2*n_sample), ("delete_mark", 3*n_sample),
                    ("left_curly_brace", 6*n_sample), ("left_sq_bracket", 3*n_sample), ("pigtail", 4*n_sample), ("question_mark", 4*n_sample),
                    ("rectangle", 4*n_sample), ("right_curly_brace", 6*n_sample), ("right_sq_bracket", 3*n_sample), ("star", 5*n_sample),
                    ("triangle", 3*n_sample), ("v", 2*n_sample), ("x", 3*n_sample)
                    }#('zig_zag', 5*n_sample)
    dataset_factory(list_gesture, Config.baseDir+'deictic/1dollar-dataset/raw/', Config.baseDir+'deictic/1dollar-dataset/resampled/')

# MDollar
if mode == 3:
    # Name gesture, n samples and strokes
    list_gesture = {("arrowhead", n_sample, 2), ("asterisk", n_sample, 3), ("D", n_sample, 2), ("exclamation_point", n_sample, 2),
                    ("H", n_sample, 3), ("half_note", n_sample, 2),
                    ("I", n_sample, 3), ("N", n_sample, 3), ("null", n_sample, 2),
                    ("P", n_sample, 2), ("pitchfork", n_sample, 2), ("six_point_star", n_sample, 2),
                    ("T", n_sample, 2), ("X", n_sample, 2)
                    }#("line", n_sample, 1), ("five_point_star", n_sample, 1, True),
    dataset_factory(list_gesture, Config.baseDir+'deictic/mdollar-dataset/raw/', Config.baseDir+'deictic/mdollar-dataset/resampled/', unistroke_mode=False)

########################################## Synthetic Dataset ##########################################################
# Sinthetic Database 1Dollar
if mode == 4:
    list_gesture = ['arrow', 'caret', 'circle', 'check', 'delete_mark', 'left_curly_brace', 'left_sq_bracket', 'pigtail',
                    'question_mark', 'rectangle', 'right_curly_brace', 'right_sq_bracket', 'star', 'triangle',
                    'v', 'x']
    list_gesture = ['question_mark', 'x']
    synthetic_dataset_factory(list_gesture, Config.baseDir+'deictic/1dollar-dataset/resampled/',
                              Config.baseDir+'deictic/1dollar-dataset/synthetic/',
                              iter=1, type='unistroke', operator = [0])#int(random.uniform(0, len(list)-1)))

if mode == 5:
    # Sinthetic Database MDollar
    list_gesture = ['arrowhead', 'asterisk', 'D', 'exclamation_point', 'H', 'half_note', 'I',
                    'N', 'null', 'P', 'pitchfork', 'six_point_star', 'T', 'X']
    synthetic_dataset_factory(list_gesture, Config.baseDir+'deictic/mdollar-dataset/resampled/',
                              Config.baseDir+'deictic/mdollar-dataset/synthetic/',
                              iter=25, type='multistroke', operator=[2])#int(random.uniform(1, len(list)-1)))
########################################## Ten-Cross-Validation Dataset ##########################################################
# todo: is it still useful?
# Ten-Cross-Validation 1Dollar
if mode == 6:
    list_gesture = ['arrow', 'caret', 'circle', 'check', 'delete_mark', 'left_curly_brace', 'left_sq_bracket', 'pigtail',
                    'question_mark', 'rectangle', 'right_curly_brace', 'right_sq_bracket', 'star', 'triangle',
                    'v', 'x']
    ten_cross_validation_factory(list_gesture, Config.baseDir+'deictic/1dollar-dataset/resampled/',
                                 Config.baseDir+'deictic/1dollar-dataset/ten-cross-validation/')
    # Ten-Cross-Validation MDollar
if mode == 7:
    list_gesture = ['arrowhead', 'asterisk', 'D', 'exclamation_point', 'H', 'half_note', 'I',
                    'N', 'null', 'P', 'pitchfork', 'six_point_star', 'T', 'X']
    # path folders
    datasetDir = baseDir + "deictic/mdollar-dataset/resampled/"
    reportDir = baseDir + "deictic/mdollar-dataset/ten-cross-validation/"
    outputDir = reportDir
    # For each gesture in the dataset
    for gesture in list_gesture:
        # Reads the report (this report shows the link model-file for each file)
        file_array = [] # contains all file in the gesture dataset
        labels = [] # contains the model that recognized each file
        # Extracts files and labels
        with open(reportDir+gesture+'/report.csv', "r") as f:
            reader = csv.reader(f, delimiter=',')
            vals = list(reader)
            for row in vals:
                new_row = []
                new_row.append(row[0])

                v = int(row[1])
                if v<0:
                    v=0
                new_row.append(v)

                file_array.append(new_row)
                labels.append(v)

        # Computes rate
        percent_rates = numpy.bincount(labels)
        rates = []
        for value in percent_rates:
            new_row = []
            rate = (value*100)/len(file_array)
            new_row.append(value)
            new_row.append(rate)
            rates.append(new_row)

        # Define new ten-cross-validation dataset
        dataset = CsvDataset(datasetDir+gesture+'/')
        for i in range(0, 10):
            dataset.ten_cross_validation(outputDir+gesture+'/', i, rates, file_array)
            #ten_cross_validation_factory(list, baseDir+'deictic/mdollar-dataset/resampled/', baseDir+'deictic/mdollar-dataset/ten-cross-validation/')
########################################## 1-dollar parsed primitives ##########################################################
from real_time.tree import *
if mode == 8:
    gesture_dataset = {
        'arrow': [CsvDatasetExtended(Config.baseDir+"Tree_test/arrow/")],
        'caret': [CsvDatasetExtended(Config.baseDir+"Tree_test/caret/")],
        'check': [CsvDatasetExtended(Config.baseDir+"Tree_test/check/")],
        'circle': [CsvDatasetExtended(Config.baseDir+"Tree_test/circle/")],
        'delete_mark': [CsvDatasetExtended(Config.baseDir+"Tree_test/delete_mark/")],
        # 'left_curly_brace': [CsvDatasetExtended(Config.baseDir+"Tree_test/left_curly_brace/")],
        'left_sq_bracket': [CsvDatasetExtended(Config.baseDir+"Tree_test/left_sq_bracket/")],
        # 'pigtail': [CsvDatasetExtended(Config.baseDir+"Tree_test/pigtail/")],
        # 'question_mark': [CsvDatasetExtended(Config.baseDir+"Tree_test/question_mark/")],
        'rectangle': [CsvDatasetExtended(Config.baseDir+"Tree_test/rectangle/")],
        # 'right_curly_brace': [CsvDatasetExtended(Config.baseDir+"Tree_test/right_curly_brace/")],
        'right_sq_bracket': [CsvDatasetExtended(Config.baseDir+"Tree_test/right_sq_bracket/")],
        'star': [CsvDatasetExtended(Config.baseDir+"Tree_test/star/")],
        'triangle': [CsvDatasetExtended(Config.baseDir+"Tree_test/triangle/")],
        'v': [CsvDatasetExtended(Config.baseDir+"Tree_test/v/")],
        'x': [CsvDatasetExtended(Config.baseDir+"Tree_test/x/")]
    }
    primitives = readChangePrimitivesFile(Config.baseDir+'Tree_test/manualRecognition/changePrimitives.csv')
    for key,tuple in gesture_dataset.items():
        dataset = tuple[1][0]
        # add new column #
        for file in dataset.readDataset():
            # adapt last marker
            value = primitives[file.filename]
            temp = [(value[0],0)]
            for index_item in range(1,len(value)):
                amount = value[index_item]-value[index_item-1]
                temp.append((amount,index_item))
            temp.append((len(file.points)-value[index_item],index_item))
            primitives[file.filename]=temp
            # concatenate
            new_column = numpy.concatenate([numpy.full(shape=(dim[0],1),fill_value=dim[1])
                                            for dim in primitives[file.filename]])
            file.points = numpy.column_stack([file.points, new_column])
            file.save(Config.baseDir+'deictic/1dollar-dataset/parsed/'+key+'/')


