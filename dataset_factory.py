from dataset import *
from gesture import *
import random

random.seed()
def synthetic_dataset_factory(inputBase, outputBase, names, iter):
    # Get all gesture's dataset
    list_dataset = []
    for name in names:
        list_dataset.append(CsvDataset(inputBase+name+'/'))

    for index in range(0, iter):

        num_rand_1 = int(random.uniform(0, len(list_dataset)-1))#
        num_rand_2 = int(random.uniform(0, len(list_dataset)-1))#
        if(num_rand_1 == num_rand_2):#
            num_rand_2 = (num_rand_2 + 1) % len(list_dataset)-1#

        # Choice
        #if (operator == Operator.choice):
        #    operator = 'choice'

        # Iterative
        operator = 1
        if (operator == 1):
            filename = 'iterative_'+names[num_rand_1]
            if not os.path.exists(outputBase + '/' + filename):
                os.makedirs(outputBase+'/' + filename)
            # Crea sequenze
            MergeIterativeDataset.create_iterative_dataset(list_dataset[num_rand_1], outputBase+'/'+filename+'/'+filename)
            print('dataset ' + filename + ' created')

        # Disabling : iterative + ground
        operator = 0
        if (operator == 0):
            # Crea sequenze
            filename = 'disabling_'+names[num_rand_1] +'_'+ names[num_rand_2]
            if not os.path.exists(outputBase + filename):
                os.makedirs(outputBase + filename)
            MergeDisablingDataset.create_disabling_dataset([CsvDataset(outputBase+'iterative_'+names[num_rand_1]+'/'), list_dataset[num_rand_2]],
                                                           outputBase+filename+'/'+filename)
            print('dataset ' + filename + ' created')


        # Parallel
        operator = 2
        if (operator == 2):
            list = []
            list.append(list_dataset[num_rand_1])
            list.append(list_dataset[num_rand_2])
            filename = 'parallel_'+names[num_rand_1] +'_'+ names[num_rand_2]
            if not os.path.exists(outputBase + filename):
                os.makedirs(outputBase + filename)
            #filename = ''
            #for i in range(0, int(random.uniform(1, len(list_dataset)))):
            #    num_rand = int(random.uniform(0, len(list_dataset)-1))
            #    filename = 'parallel_'+filename + '_' + names[num_rand]
            #    list.append(list_dataset[num_rand])
            # Make sequences
            MergeParallelDataset.create_parallel_dataset(list, outputBase+filename+'/'+filename, flag_trasl=False)
            print('dataset ' + filename + ' created')

        # Sequence
        operator = 3
        if (operator == 3):
            list = []
            filename = 'sequence_'+names[num_rand_1] +'_'+ names[num_rand_2]
            list.append(list_dataset[num_rand_1])
            list.append(list_dataset[num_rand_2])
            if not os.path.exists(outputBase + filename):
                os.makedirs(outputBase + filename)
            #for i in range(0, int(random.uniform(1, len(list_dataset)))):
            #    num_rand = int(random.uniform(0, len(list_dataset)-1))
            #    filename = 'sequence_'+filename + '_' + names[num_rand]
            #    list.append(list_dataset[num_rand])
            MergeSequenceDataset.create_sequence_dataset(list, outputBase+filename+'/'+filename)
            print('dataset ' + filename + ' created')


def dataset_factory(list, inputDir, outputDir, unistroke_mode = True):

    for gesture in list:
        input_dir = inputDir+gesture[0]+'/'
        output_dir = outputDir+gesture[0]+'/'
        dataset = CsvDataset(input_dir)

        # Transform
        if unistroke_mode:
            transform1 = NormaliseLengthTransform(axisMode=True)
        else:
            transform1 = NormaliseLengthTransform(axisMode=gesture[3])
        transform2 = ScaleDatasetTransform(scale=100)
        transform3 = CenteringTransform()
        #transform4 = RotateCenterTransform(traslationMode=True)
        if unistroke_mode:
            transform5 = ResampleInSpaceTransform(samples=gesture[1])
        else:
            transform5 = ResampleInSpaceTransformMultiStroke(samples=gesture[1], strokes = gesture[2])
        # Apply transforms
        dataset.addTransform(transform1)
        dataset.addTransform(transform2)
        dataset.addTransform(transform3)
        #dataset.addTransform(transform4)
        dataset.addTransform(transform5)

        dataset.applyTransforms(output_dir)
    return


baseDir = '/home/alessandro/PycharmProjects/deictic/repository/'
#baseDir  = '/Users/davide/Google Drive/Dottorato/Software/python/hmmtest/repository/'

mode = 5
n_sample = 40

# Unica
if mode == 1:
    list = {("rectangle", 4*n_sample), ("triangle", 3*n_sample), ("caret", 2*n_sample), ("v", 2*n_sample), ("x", 3*n_sample),
            ("square-braket-left", 3*n_sample), ("square-braket-right", 3*n_sample), ("delete", 3*n_sample),
            ("star", 5*n_sample)
            }#, ("left", n_sample), ("right", n_sample)
    dataset_factory(list, baseDir+'deictic/unica-dataset/raw/', baseDir+'deictic/unica-dataset/resampled/')

# 1Dollar
if mode == 2:
    # name and n_samples
    list = {("arrow", 4*n_sample), ("caret", 2*n_sample), ("circle", 4*n_sample), ("check", 2*n_sample), ("delete_mark", 3*n_sample),
            ("left_curly_brace", 6*n_sample), ("left_sq_bracket", 3*n_sample), ("pigtail", 4*n_sample), ("question_mark", 4*n_sample),
            ("rectangle", 4*n_sample), ("right_curly_brace", 6*n_sample), ("right_sq_bracket", 3*n_sample), ("star", 5*n_sample),
            ("triangle", 3*n_sample), ("v", 2*n_sample), ("x", 3*n_sample)
            }
    dataset_factory(list, baseDir+'deictic/1dollar-dataset/raw/', baseDir+'deictic/1dollar-dataset/resampled/')

# MDollar
if mode == 3:
    # Name gesture, n samples and strokes
    list = {("arrowhead", n_sample, 2, True), ("asterisk", n_sample, 3, True), ("D", n_sample, 2, True), ("exclamation_point", n_sample, 2, False),
            ("five_point_star", n_sample, 1, True), ("H", n_sample, 3, True), ("half_note", n_sample, 2, True),
            ("I", n_sample, 3, True), ("N", n_sample, 3, True), ("null", n_sample, 2, True),
            ("P", n_sample, 2, True), ("pitchfork", n_sample, 2, True), ("six_point_star", n_sample, 2, True),
            ("T", n_sample, 2, True), ("X", n_sample, 2, True)
            }#("line", n_sample, 1)
    dataset_factory(list, baseDir+'deictic/mdollar-dataset/raw/', baseDir+'deictic/mdollar-dataset/resampled/', unistroke_mode=False)

# Sinthetic Database 1Dollar
if mode == 4:
    list = ['arrow', 'caret', 'circle', 'check', 'delete_mark', 'left_curly_brace', 'left_sq_bracket', 'pigtail',
            'question_mark', 'rectangle', 'right_curly_brace', 'right_sq_bracket', 'star', 'triangle',
            'v', 'x']
    synthetic_dataset_factory(baseDir+'deictic/1dollar-dataset/resampled/', baseDir+'deictic/1dollar-dataset/resampled/',
                              list, iter=16)#int(random.uniform(0, len(list)-1)))

if mode == 5:
    # Sinthetic Database MDollar
    list = ['arrowhead', 'asterisk', 'D', 'exclamation_point', 'H', 'half_note', 'I',
            'N', 'null', 'P', 'pitchfork', 'six_point_star', 'T', 'X']
    synthetic_dataset_factory(baseDir+'deictic/mdollar-dataset/resampled/', baseDir+'deictic/mdollar-dataset/resampled/',
                              list, iter=16)#int(random.uniform(1, len(list)-1)))

## Original
# Unica
if mode == 10:
    inputDir = baseDir + 'original/unica-dataset/'
    outputDir = baseDir + 'deictic/unica-dataset/raw'
    converter = UnicaConverter()
    converter.create_deictic_dataset(inputDir, outputDir)
# 1Dollar
if mode == 11:
    inputDir = baseDir + 'original/1dollar-dataset/'
    outputDir = baseDir + 'deictic/1dollar-dataset/raw'
    converter = Dollar1Converter()
    converter.create_deictic_dataset(inputDir, outputDir)
# MDollar
if mode == 12:
    inputDir = baseDir + 'original/mdollar-dataset'
    outputDir = baseDir + 'deictic/mdollar-dataset/raw'
    converter = DollarMConverter()
    converter.order_files(inputDir, inputDir)
    converter.create_deictic_dataset(inputDir, outputDir)




