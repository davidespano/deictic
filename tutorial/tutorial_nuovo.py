from gesture import Point, Line
from test import *

def firstExample():
    '''
        shows how to describe a gesture with Deictic and create its model.
    '''
    expressions = {
        'swipe_right':[Point(0,0)+Line(3,0)]
    }
    # create deictic model for swipe_right
    hmm_swipe_right = CreateRecognizer.generateHMMs(expressions = expressions, num_states = 6, spu = 20)
    print(hmm_swipe_right)

def secondExample():
    '''
        secondExample, starting from firstExample, shows how to define a set of CsvDataset,
        how to compare two different gestures and, finally, how to plot results.
    '''
    # describe gesture swipe_right through deictic
    gesture_expressions = {
        'swipe_right':[Point(0,0)+Line(3,0)],
        'swipe_left': [Point(3,0)+Line(-3, 0)]
    }
    # get swipe right and swipe left datasets
    gesture_dataset = {
        'swipe_right':[CsvDatasetExtended(Config.baseDir+'repository/deictic/unica-dataset/right/')],
        'swipe_left': [CsvDatasetExtended(Config.baseDir + 'repository/deictic/unica-dataset/left/')]
    }
    # create deictic model for swipe right and left
    gesture_hmms = CreateRecognizer.generateHMMs(gesture_expressions)
    # start log-probability-based test
    result = Test.offlineTest(gesture_hmms=gesture_hmms, gesture_datasets=gesture_dataset)
    # show result through confusion matrix
    result.plot()
    # save result on csv file
    print(result.log_probabilities)

def thirdExample():
    '''

    :return:
    '''
    # get deictic expressions which describe 1$ unistroke dataset
    gesture_expressions = DatasetExpressions.returnExpressions(selected_dataset=TypeDataset.unistroke_1dollar)
    # create hmms
    gesture_hmms = CreateRecognizer.generateHMMs(expressions=gesture_expressions)
    # get sequence test (by using the first model of circle for generating a sample)
    sequence_test = gesture_hmms['circle'][0].sample()
    # compare hmms and show the computed log probabilities for each gesture
    log_probabilities = Test.compare(sequence=sequence_test, gesture_hmms=gesture_hmms)
    #print("The gesture with the highest log probabilities value is " +index_label)
    print(log_probabilities)

def fourthExample():
    '''

    :return:
    '''
    # get expressions which describe 1$ unistroke dataset
    gesture_expressions = DatasetExpressions.returnExpressions(selected_dataset=TypeDataset.unistroke_1dollar)
    # create models using 6 states for each ground term
    gesture_hmms = CreateRecognizer.generateHMMs(expressions=gesture_expressions, num_states=6, spu=20)
    # get 1$ unistroke dataset
    gesture_dataset = DatasetFolders.returnFolders(selected_dataset=TypeDataset.unistroke_1dollar)
    # start log-probability-based test
    results = Test.offlineTest(gesture_hmms=gesture_hmms,gesture_datasets=gesture_dataset)
    # show result through confusion matrix
    results.confusion_matrix.plot()

def fifthExample():
    '''

    :return:
    '''
    # get expressions which describe 1$ multistroke dataset
    gesture_expressions = DatasetExpressions.returnExpressions(selected_dataset=TypeDataset.multistroke_1dollar)
    # create models using 6 states for each ground term
    gesture_hmms = CreateRecognizer.generateHMMs(expressions=gesture_expressions, num_states=6, spu=20)
    # get 1$ multistroke dataset
    gesture_dataset = DatasetFolders.returnFolders(selected_dataset=TypeDataset.multistroke_1dollar)
    # start log-probability-based test
    results = Test.offlineTest(gesture_hmms=gesture_hmms, gesture_datasets=gesture_dataset)
    # show result through confusion matrix
    results.confusion_matrix.plot()

def sixthExample():
    '''
        G-gene
    :return:
    '''
    pass

def seventhExample():
    '''

    :return:
    '''
    # gesture expressions
    gesture_expressions = DatasetExpressions.returnExpressions(selected_dataset=TypeDataset.unistroke_1dollar)
    # create gesture recognizers
    tree = Tree(gesture_exp=gesture_expressions)

    # gesture dataset
    gesture_dataset = {
        'arrow': (4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/primitives/arrow/")]),
        'caret': (2,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/primitives/caret/")]),
        #'check': (2,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/primitives/check/")]),
        #'circle':(4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/primitives/circle/")]),
        'delete_mark': (3,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/primitives/delete_mark/")]),
        #'left_curly_brace': (6,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/parsed/left_curly_brace/")]),
        'left_sq_bracket': (3,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/primitives/left_sq_bracket/")]),
        #'pigtail': (4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/parsed/pigtail/")]),
        #'question_mark': (4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/parsed/question_mark/")]),
        'rectangle': (4,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/primitives/rectangle/")]),
        #'right_curly_brace': (6,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/parsed/right_curly_brace/")]),
        'right_sq_bracket': (3,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/primitives/right_sq_bracket/")]),
        'star': (5,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/primitives/star/")]),
        'triangle': (3, [CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/primitives/triangle/")]),
        'v': (2,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/primitives/v/")]),
        'x': (3,[CsvDatasetExtended(Config.baseDir+"deictic/1dollar-dataset/primitives/x/")])
    }


    # start test
    for perc in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]:
        results = Test.onlineTest(gesture_tree=tree,
                                  gesture_datasets=gesture_dataset,
                                  perc_completed=perc,
                                  best_hmm=3)
        results.confusion_matrix.plot(title="Online Recognition - "+str(perc)+'%')





# main #
print('First Example:')
firstExample()
print('Second Example:')
secondExample()
print('Third Example:')
thirdExample()
print('Fourth Example')
fourthExample()
print('Fifth Example')
fifthExample()
print('Sixth Example')
sixthExample()
print('Seventh Example')
seventhExample()