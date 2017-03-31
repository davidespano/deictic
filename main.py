from dataset import *
from gesture import *
from test import *



# Main
baseDir = '/home/alessandro/PycharmProjects/deictic/repository/'
n_states = 6 # Numero stati
n_samples = 40
iterations = 1 # k-fold cross-validation
mode = 7


#baseDir  = '/Users/davide/Google Drive/Dottorato/Software/python/hmmtest/repository/'
baseDir = '/Users/davide/PycharmProjects/deictic/repository/'
trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'
testDir = baseDir + "deictic/unica-dataset/resampled/"

################################################################ DEICTIC HMM ################################################################
if mode in [-1, 0, 1]:
    # Unica
    if mode == -1:
        folders = ['caret', 'check', 'delete_mark', 'left_sq_bracket', 'right_sq_bracket',
                   'star', 'triangle', 'v', 'x']
        gestureDir = baseDir + 'deictic/unica-dataset/resampled/'
        type = 'unica-'
    # 1Dollar
    elif mode == 0:
        folders = ['arrow', 'caret', 'check', 'circle', 'delete_mark', 'left_curly_brace', 'left_sq_bracket',
                   'pigtail', 'question_mark', 'rectangle', 'right_curly_brace', 'right_sq_bracket',
                   'star', 'triangle', 'v', 'x']
        gestureDir = baseDir + 'deictic/1dollar-dataset/resampled/'
        type = 'unistroke-'
    # MDollar
    else:
        folders = ['D', 'H', 'I', 'N', 'P', 'T', 'X', 'arrowhead',
                   'asterisk', 'exclamation_point', 'half_note', 'null', 'pitchfork',
                   'six_point_star']
        gestureDir = baseDir + 'deictic/mdollar-dataset/resampled/'
        type = 'multistroke-'

    hmms = []
    parse = Parse(n_states, n_samples)
    for folder in folders:
        model = parse.parse_expression(type+folder)
        hmms.append(model)

    t = test(hmms, gestureDir, folders, plot=False)
    results = t.all_files()

    ## Plot
    #for hmm in hmms:
    #    hmm.plot()

############################################################ DEICTIC Synthetic HMM ###########################################################
if mode in [2,3]:
    if mode == 2:
        gestureDir = baseDir + 'deictic/1dollar-dataset/synthetic/'
    else:
        gestureDir = baseDir + 'deictic/mdollar-dataset/synthetic/'

    folders = [name for name in os.listdir(gestureDir) if os.path.isdir(os.path.join(gestureDir, name))]
    hmms = []
    parse = Parse(n_states, n_samples)
    for folder in folders:
        model = parse.parse_expression(folder)
        hmms.append(model)

    t = test(hmms, gestureDir, folders, plot=False)
    results = t.all_files()

############################################################ DEICTIC Ten-Cross-Validation HMM ###########################################################
if mode in [4,5]:
    if mode == 4:
        folders = ['arrow', 'caret', 'check', 'circle', 'delete_mark', 'left_curly_brace', 'left_sq_bracket',
                   'pigtail', 'question_mark', 'rectangle', 'right_curly_brace', 'right_sq_bracket',
                   'star', 'triangle', 'v', 'x']
        gestureDir = baseDir + 'deictic/1dollar-dataset/resampled/'
        list_filesDir = baseDir + 'deictic/1dollar-dataset/ten-cross-validation'
        type = 'unistroke-'
    else:
        folders = ['D', 'H', 'I', 'N', 'P', 'T', 'X', 'arrowhead',
                   'asterisk', 'exclamation_point', 'half_note', 'null', 'pitchfork',
                   'six_point_star']
        gestureDir = baseDir + 'deictic/mdollar-dataset/resampled/'
        list_filesDir = baseDir + 'deictic/mdollar-dataset/ten-cross-validation'
        type = 'multistroke-'

    hmms = []
    parse = Parse(n_states, n_samples)
    for folder in folders:
        model = parse.parse_expression(type+folder)
        hmms.append(model)

    t = test(hmms, gestureDir, folders, plot=False)
    results = t.ten_cross_validation(list_filesDir)


################################################################ ADHOC HMM ################################################################
if mode in [6, 7]:
    ## Adhoc hmm - 1Dollar
    if mode == 6:
        list_gesture = [("rectangle", n_states*4), ("triangle", n_states*3), ("caret", n_states*2), ("v", n_states*2), ("x", n_states*3),
                    ("left_sq_bracket", n_states*3), ("right_sq_bracket", n_states*3), ("delete_mark", n_states*4), ("star", n_states*4),
                    ("arrow", n_states*4), ("check", n_states*2), ("circle", n_states*4), ("left_curly_brace", n_states*6),
                    ("right_curly_brace", n_states*6), ("pigtail", n_states*4), ("question_mark", n_states*4)]
        list_gesture = [("caret", n_states*2), ("v", n_states*2)]
        gestureDir = baseDir + 'deictic/1dollar-dataset/resampled/'
        list_filesDir = baseDir + 'deictic/1dollar-dataset/ten-cross-validation'
        n_features = 2

        gestures = [i[0] for i in list_gesture]
        results = None

        # Create hmm gesture, training and testing sequences
        hmms = []

        for k in range(0, iterations):
            for gesture in list_gesture:
                # Training dataset
                training_dataset = CsvDataset(gestureDir+gesture[0]+'/'). \
                    read_ten_cross_validation_dataset(list_filesDir + gesture +'/', type='train')
                # Create and training hmm
                hmms.append(create_hmm_gesture(gesture[0], training_dataset, gesture[1], n_features))

            t = test(hmms, gestureDir, gestures, plot=False, results=results)
            results = t.ten_cross_validation(list_filesDir, iterations=10)
            print("K = {}".format(k))
            print(results)

    ## Adhoc hmm - MDollar
    if mode == 7:
        #list_gesture = [("D", n_states*3), ("H", n_states*3), ("I", n_states*3), ("N", n_states*3), ("P", n_states*3),
        #                ("T", n_states*2), ("X", n_states*2), ("arrowhead", n_states*3), ("asterisk", n_states*4),
        #                ("exclamation_point", n_states*2), ("half_note", n_states*3),
        #                ("null", n_states*5), ("pitchfork", n_states*3), ("six_point_star", n_states*6)]
        list_gesture = [("D", n_states*3), ("H", n_states*3)]
        gestureDir = baseDir + 'deictic/mdollar-dataset/resampled/'
        list_filesDir = baseDir + 'deictic/mdollar-dataset/ten-cross-validation/'
        n_features = 3

        # Training
        gestures = dict()

        for k in range(0, iterations):
            for gesture in list_gesture:
                hmms = []
                # Read training dataset
                training_dataset = CsvDataset(gestureDir+gesture[0]+'/')
                files = open(list_filesDir + gesture[0] +'/train_ten-cross-validation_{}.txt'.format(str(k))).readlines()
                files = files[0].split('/')
                # Order training dataset
                list_dataset_for_models = []
                list_files = []
                list_occurrence_models = []
                for file in files:
                    file = file.split('_')
                    list_files.append(file[0])
                    list_occurrence_models.append(int(file[1]))

                n_models = max(list_occurrence_models)
                for i in range(0, n_models+1):
                    list_dataset_for_models.append([]);

                for i in range(0, len(list_occurrence_models)):

                    list_dataset_for_models[list_occurrence_models[i]].append(training_dataset.read_file(list_files[i]));

                for i in range(0, len(list_dataset_for_models)):
                    # Create and training hmm
                    hmms.append(create_hmm_gesture(gesture[0], list_dataset_for_models[i], gesture[1], n_features))
                gestures[gesture].append(hmms)

            results = compares_deictic_models(hmms, gestureDir)
            print("K = {}".format(k))
            print(results)


# Stampa matrice di confusione
print(results)


if mode == 24:
    #t = Point(0,0) + Line(4,0) + Point(2,0) + Line(0, -4)
    #t = Point(0,0) + Line(6,4) + Line(-4,0) + Line(5,1) + Line(-1, -4)
    # t.plot()
    gesture_models[10][0].plot()
    factory = ClassifierFactory()
    factory.setLineSamplesPath(trainingDir)
    factory.setClockwiseArcSamplesPath(arcClockWiseDir)
    factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
    model, edges = factory.createClassifier(gesture_models[10][0])
    plot_gesture(model)

if mode == 25:
    dataset = CsvDataset(testDir + "delete_mark/")
    #dataset.plot()
    dataset.plot(singleMode=True)

if mode == 26:
    d = NormalDistribution(2.0, 0.5)
    samples = [d.sample() for i in range(10000)]
    plt.hist(samples, edgecolor='c', color='c', bins=50)
    plt.show()