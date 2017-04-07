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
testDir = baseDir + "deictic/mdollar-dataset/resampled/"

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
        list_gesture = [("D", 2),
                        #("H", 3),
                        #("I", 3),
                        #("N", 3),
                        ("P", 2),
                        #("T", 2),
                        #("X", 2),
                        #("arrowhead", 2),
                        #("asterisk", 3),
                        #("exclamation_point", 2),
                        #("half_note", 3),
                        #("null", 2),
                        #("pitchfork", 2),
                        #("six_point_star", 2)
                        ]
        #list_gesture = [("D", 2), ("X", 2)]

        list_avoid = {
            "D" : [2,3], "H" : [], "I": [], "N" : [], "P": [0], "T": [1,2,3], "X": [], "arrowhead": [], "asterisk" : [],
            #"exclamation_point": [], "half_note": [],
            "null": [0], "pitchfork":[0],
            #"six_point_star": []
        }

        gestureDir = baseDir + 'deictic/mdollar-dataset/resampled/'
        list_filesDir = baseDir + 'deictic/mdollar-dataset/ten-cross-validation/'
        n_features = 3

        # Training
        gestures = dict()

        for k in range(0, iterations):
            for gesture in list_gesture:
                hmms = []
                gestures[gesture[0]] = []
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
                    print("---------------  gesture: {0} model: {1} -----------------".format(gesture[0], i));

                    if not i in list_avoid[gesture[0]]:
                        hmms.append(
                            create_hmm_gesture(
                                gesture[0],
                                list_dataset_for_models[i],
                                gesture[1] * n_states, n_features,
                                weights = [1,1, 10000],
                                stroke = gesture[1])
                            )
                gestures[gesture[0]] = hmms

            print("inizio il test")
            results = compares_deictic_models(gestures, gestureDir)
            print("K = {}".format(k))
            print(results)

if mode == 8:
    folders = ['v', 'caret', 'left_sq_bracket', 'right_sq_bracket', 'x', 'delete_mark',
               'triangle', 'rectangle']
    gestureDir = baseDir + 'deictic/unica-dataset/resampled/'

    confusion = numpy.zeros((len(folders), len(folders)))
    for k in range(0, 13):
        hmms = []
        test = []
        for dir in folders:
            data = CsvDataset(gestureDir + dir + '/');
            one, train = data.leave_one_out(leave_index=k);
            test.append(one)
            print("---------------  gesture: {0} fold: {1} -----------------".format(dir, k));
            model = create_hmm_gesture(dir, train, n_states, 2)
            model.fit(train, use_pseudocount=True)
            hmms.append(model)
        for i in range(0, len(test)):
            max_norm_log_probability = -sys.maxsize
            j = 0
            for h in range(0, len(hmms)):
                normLog = hmms[h].log_probability(test[i]) / len(test[i])

                if normLog > max_norm_log_probability:
                    j = h
                    max_norm_log_probability = normLog

            confusion[i][j] = confusion[i][j] + 1
    print(folders)
    print(confusion)





# Stampa matrice di confusione
#print(results)


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
    dataset = CsvDataset(testDir + "D/")
    #dataset.plot()
    sampleNames0 =[
        '66-finger-slow-D-04.csv',
        '85-finger-slow-D-07.csv',
        '98-stylus-slow-D-09.csv'
    ];



    sampleNames1 = [
        '10-stylus-fast-D-01.csv',
        '10-stylus-fast-D-02.csv',
        '10-stylus-fast-D-03.csv',
'10-stylus-fast-D-04.csv',
'10-stylus-fast-D-05.csv',
'10-stylus-fast-D-06.csv',
'10-stylus-fast-D-07.csv',
'10-stylus-fast-D-08.csv',
'10-stylus-fast-D-09.csv',
'10-stylus-fast-D-10.csv',
'10-stylus-medium-D-01.csv',
'10-stylus-medium-D-02.csv',
'10-stylus-medium-D-03.csv',
'10-stylus-medium-D-04.csv',
'10-stylus-medium-D-05.csv',
'10-stylus-medium-D-06.csv',
'10-stylus-medium-D-07.csv',
'10-stylus-medium-D-08.csv',
'10-stylus-medium-D-09.csv',
'10-stylus-medium-D-10.csv',
'10-stylus-slow-D-01.csv',
'10-stylus-slow-D-03.csv',
'10-stylus-slow-D-04.csv',
'10-stylus-slow-D-06.csv',
'10-stylus-slow-D-07.csv',
'10-stylus-slow-D-08.csv',
'10-stylus-slow-D-09.csv',
'10-stylus-slow-D-10.csv',
'11-stylus-fast-D-01.csv',
'11-stylus-fast-D-03.csv',
'11-stylus-fast-D-04.csv',
'11-stylus-fast-D-06.csv',
'11-stylus-fast-D-07.csv',
'11-stylus-fast-D-09.csv',
'11-stylus-fast-D-10.csv',
'11-stylus-medium-D-01.csv',
'11-stylus-medium-D-02.csv',
'11-stylus-medium-D-03.csv',
'11-stylus-medium-D-05.csv',
'11-stylus-medium-D-07.csv',
'11-stylus-medium-D-08.csv',
'11-stylus-medium-D-09.csv',
'11-stylus-slow-D-01.csv',
'11-stylus-slow-D-02.csv',
'11-stylus-slow-D-03.csv',
'11-stylus-slow-D-04.csv',
'11-stylus-slow-D-05.csv',
'11-stylus-slow-D-06.csv',
'11-stylus-slow-D-07.csv',
'11-stylus-slow-D-08.csv',
'11-stylus-slow-D-10.csv',
'12-stylus-fast-D-01.csv',
'12-stylus-fast-D-02.csv',
'12-stylus-fast-D-03.csv',
'12-stylus-fast-D-04.csv',
'12-stylus-fast-D-05.csv',
'12-stylus-fast-D-06.csv',
'12-stylus-fast-D-07.csv',
'12-stylus-fast-D-08.csv',
'12-stylus-fast-D-09.csv',
'12-stylus-fast-D-10.csv',
'12-stylus-medium-D-01.csv',
'12-stylus-medium-D-02.csv',
'12-stylus-medium-D-03.csv',
'12-stylus-medium-D-04.csv',
'12-stylus-medium-D-05.csv',
'12-stylus-medium-D-06.csv',
'12-stylus-medium-D-07.csv',
'12-stylus-medium-D-08.csv',
'12-stylus-medium-D-09.csv',
'12-stylus-medium-D-10.csv',
'12-stylus-slow-D-01.csv',
'12-stylus-slow-D-02.csv',
'12-stylus-slow-D-03.csv',
'12-stylus-slow-D-04.csv',
'12-stylus-slow-D-05.csv',
'12-stylus-slow-D-06.csv',
'12-stylus-slow-D-07.csv',
'12-stylus-slow-D-08.csv',
'12-stylus-slow-D-09.csv',
'12-stylus-slow-D-10.csv',
'22-stylus-fast-D-02.csv',
'22-stylus-fast-D-03.csv',
'22-stylus-fast-D-04.csv',
'22-stylus-fast-D-05.csv',
'22-stylus-fast-D-06.csv',
'22-stylus-fast-D-07.csv',
'22-stylus-fast-D-08.csv',
'22-stylus-fast-D-10.csv',
'22-stylus-medium-D-02.csv',
'22-stylus-medium-D-03.csv',
'22-stylus-medium-D-04.csv',
'22-stylus-medium-D-05.csv',
'22-stylus-medium-D-07.csv',
'22-stylus-medium-D-08.csv',
'22-stylus-medium-D-09.csv',
'22-stylus-medium-D-10.csv',
'22-stylus-slow-D-01.csv',
'22-stylus-slow-D-02.csv',
'22-stylus-slow-D-03.csv',
'22-stylus-slow-D-04.csv',
'22-stylus-slow-D-05.csv',
'22-stylus-slow-D-06.csv',
'22-stylus-slow-D-07.csv',
'22-stylus-slow-D-08.csv',
'22-stylus-slow-D-09.csv',
'22-stylus-slow-D-10.csv',
'28-stylus-fast-D-01.csv',
'28-stylus-fast-D-02.csv',
'28-stylus-fast-D-03.csv',
'28-stylus-fast-D-04.csv',
'28-stylus-fast-D-05.csv',
'28-stylus-fast-D-06.csv',
'28-stylus-fast-D-07.csv',
'28-stylus-fast-D-08.csv',
'28-stylus-fast-D-09.csv',
'28-stylus-fast-D-10.csv',
'28-stylus-medium-D-01.csv',
'28-stylus-medium-D-02.csv',
'28-stylus-medium-D-03.csv',
'28-stylus-medium-D-04.csv',
'28-stylus-medium-D-05.csv',
'28-stylus-medium-D-06.csv',
'28-stylus-medium-D-07.csv',
'28-stylus-medium-D-08.csv',
'28-stylus-medium-D-09.csv',
'28-stylus-medium-D-10.csv',
'28-stylus-slow-D-01.csv',
'28-stylus-slow-D-02.csv',
'28-stylus-slow-D-03.csv',
'28-stylus-slow-D-04.csv',
'28-stylus-slow-D-05.csv',
'28-stylus-slow-D-06.csv',
'28-stylus-slow-D-07.csv',
'28-stylus-slow-D-08.csv',
'28-stylus-slow-D-09.csv',
'28-stylus-slow-D-10.csv',
'41-finger-fast-D-01.csv',
'41-finger-fast-D-02.csv',
'41-finger-fast-D-03.csv',
'41-finger-fast-D-04.csv',
'41-finger-fast-D-05.csv',
'41-finger-fast-D-07.csv',
'41-finger-fast-D-08.csv',
'41-finger-fast-D-09.csv',
'41-finger-fast-D-10.csv',
'41-finger-medium-D-01.csv',
'41-finger-medium-D-02.csv',
'41-finger-medium-D-03.csv',
'41-finger-medium-D-04.csv',
'41-finger-medium-D-05.csv',
'41-finger-medium-D-06.csv',
'41-finger-medium-D-07.csv',
'41-finger-medium-D-08.csv',
'41-finger-medium-D-09.csv',
'41-finger-medium-D-10.csv',
'41-finger-slow-D-01.csv',
'41-finger-slow-D-02.csv',
'41-finger-slow-D-05.csv',
'41-finger-slow-D-06.csv',
'41-finger-slow-D-07.csv',
'41-finger-slow-D-08.csv',
'41-finger-slow-D-09.csv',
'41-finger-slow-D-10.csv',
'58-finger-fast-D-01.csv',
'58-finger-fast-D-03.csv',
'58-finger-fast-D-04.csv',
'58-finger-fast-D-05.csv',
'58-finger-fast-D-06.csv',
'58-finger-fast-D-07.csv',
'58-finger-fast-D-08.csv',
'58-finger-fast-D-09.csv',
'58-finger-fast-D-10.csv',
'58-finger-medium-D-01.csv',
'58-finger-medium-D-02.csv',
'58-finger-medium-D-03.csv',
'58-finger-medium-D-04.csv',
'58-finger-medium-D-05.csv',
'58-finger-medium-D-06.csv',
'58-finger-medium-D-07.csv',
'58-finger-medium-D-08.csv',
'58-finger-medium-D-09.csv',
'58-finger-medium-D-10.csv',
'58-finger-slow-D-01.csv',
'58-finger-slow-D-02.csv',
'58-finger-slow-D-04.csv',
'58-finger-slow-D-05.csv',
'58-finger-slow-D-06.csv',
'58-finger-slow-D-07.csv',
'58-finger-slow-D-08.csv',
'58-finger-slow-D-09.csv',
'61-finger-fast-D-01.csv',
'61-finger-fast-D-02.csv',
'61-finger-fast-D-03.csv',
'61-finger-fast-D-05.csv',
'61-finger-fast-D-06.csv',
'61-finger-fast-D-07.csv',
'61-finger-fast-D-08.csv',
'61-finger-fast-D-09.csv',
'61-finger-fast-D-10.csv',
'61-finger-medium-D-01.csv',
'61-finger-medium-D-02.csv',
'61-finger-medium-D-03.csv',
'61-finger-medium-D-05.csv',
'61-finger-medium-D-06.csv',
'61-finger-medium-D-07.csv',
'61-finger-medium-D-08.csv',
'61-finger-medium-D-09.csv',
'61-finger-medium-D-10.csv',
'61-finger-slow-D-01.csv',
'61-finger-slow-D-02.csv',
'61-finger-slow-D-03.csv',
'61-finger-slow-D-04.csv',
'61-finger-slow-D-05.csv',
'61-finger-slow-D-06.csv',
'61-finger-slow-D-07.csv',
'61-finger-slow-D-08.csv',
'61-finger-slow-D-09.csv',
'61-finger-slow-D-10.csv',
'66-finger-fast-D-01.csv',
'66-finger-fast-D-02.csv',
'66-finger-fast-D-03.csv',
'66-finger-fast-D-04.csv',
'66-finger-fast-D-06.csv',
'66-finger-fast-D-08.csv',
'66-finger-fast-D-09.csv',
'66-finger-fast-D-10.csv',
'66-finger-medium-D-01.csv',
'66-finger-medium-D-02.csv',
'66-finger-medium-D-03.csv',
'66-finger-medium-D-04.csv',
'66-finger-medium-D-05.csv',
'66-finger-medium-D-06.csv',
'66-finger-medium-D-08.csv',
'66-finger-medium-D-09.csv',
'66-finger-medium-D-10.csv',
'66-finger-slow-D-01.csv',
'66-finger-slow-D-02.csv',
'66-finger-slow-D-03.csv',

'66-finger-slow-D-05.csv',
'66-finger-slow-D-06.csv',
'66-finger-slow-D-07.csv',
'66-finger-slow-D-08.csv',
'66-finger-slow-D-09.csv',
'66-finger-slow-D-10.csv',
'68-stylus-fast-D-01.csv',
'68-stylus-fast-D-02.csv',
'68-stylus-fast-D-03.csv',
'68-stylus-fast-D-05.csv',
'68-stylus-fast-D-07.csv',
'68-stylus-fast-D-08.csv',
'68-stylus-fast-D-09.csv',
'68-stylus-fast-D-10.csv',
'68-stylus-medium-D-02.csv',
'68-stylus-medium-D-03.csv',
'68-stylus-medium-D-04.csv',
'68-stylus-medium-D-05.csv',
'68-stylus-medium-D-06.csv',
'68-stylus-medium-D-07.csv',
'68-stylus-medium-D-08.csv',
'68-stylus-medium-D-09.csv',
'68-stylus-medium-D-10.csv',
'68-stylus-slow-D-02.csv',
'68-stylus-slow-D-03.csv',
'68-stylus-slow-D-04.csv',
'68-stylus-slow-D-05.csv',
'68-stylus-slow-D-06.csv',
'68-stylus-slow-D-08.csv',
'68-stylus-slow-D-09.csv',
'68-stylus-slow-D-10.csv',
'71-stylus-fast-D-01.csv',
'71-stylus-fast-D-02.csv',
'71-stylus-fast-D-03.csv',
'71-stylus-fast-D-04.csv',
'71-stylus-fast-D-06.csv',
'71-stylus-fast-D-07.csv',
'71-stylus-fast-D-08.csv',
'71-stylus-fast-D-09.csv',
'71-stylus-fast-D-10.csv',
'71-stylus-medium-D-01.csv',
'71-stylus-medium-D-02.csv',
'71-stylus-medium-D-03.csv',
'71-stylus-medium-D-04.csv',
'71-stylus-medium-D-05.csv',
'71-stylus-medium-D-06.csv',
'71-stylus-medium-D-07.csv',
'71-stylus-medium-D-10.csv',
'71-stylus-slow-D-01.csv',
'71-stylus-slow-D-02.csv',
'71-stylus-slow-D-04.csv',
'71-stylus-slow-D-05.csv',
'71-stylus-slow-D-06.csv',
'71-stylus-slow-D-07.csv',
'71-stylus-slow-D-08.csv',
'71-stylus-slow-D-09.csv',
'71-stylus-slow-D-10.csv',
'73-finger-fast-D-01.csv',
'73-finger-fast-D-02.csv',
'73-finger-fast-D-04.csv',
'73-finger-fast-D-05.csv',
'73-finger-fast-D-06.csv',
'73-finger-fast-D-08.csv',
'73-finger-fast-D-09.csv',
'73-finger-fast-D-10.csv',
'73-finger-medium-D-01.csv',
'73-finger-medium-D-02.csv',
'73-finger-medium-D-03.csv',
'73-finger-medium-D-04.csv',
'73-finger-medium-D-05.csv',
'73-finger-medium-D-06.csv',
'73-finger-medium-D-07.csv',
'73-finger-medium-D-08.csv',
'73-finger-medium-D-09.csv',
'73-finger-medium-D-10.csv',
'73-finger-slow-D-03.csv',
'73-finger-slow-D-04.csv',
'73-finger-slow-D-05.csv',
'73-finger-slow-D-06.csv',
'73-finger-slow-D-07.csv',
'73-finger-slow-D-08.csv',
'73-finger-slow-D-10.csv',
'75-finger-fast-D-01.csv',
'75-finger-fast-D-02.csv',
'75-finger-fast-D-03.csv',
'75-finger-fast-D-04.csv',
'75-finger-fast-D-05.csv',
'75-finger-fast-D-06.csv',
'75-finger-fast-D-07.csv',
'75-finger-fast-D-08.csv',
'75-finger-fast-D-09.csv',
'75-finger-fast-D-10.csv',
'75-finger-medium-D-01.csv',
'75-finger-medium-D-02.csv',
'75-finger-medium-D-03.csv',
'75-finger-medium-D-04.csv',
'75-finger-medium-D-05.csv',
'75-finger-medium-D-06.csv',
'75-finger-medium-D-07.csv',
'75-finger-medium-D-08.csv',
'75-finger-medium-D-09.csv',
'75-finger-medium-D-10.csv',
'75-finger-slow-D-02.csv',
'75-finger-slow-D-03.csv',
'75-finger-slow-D-04.csv',
'75-finger-slow-D-05.csv',
'75-finger-slow-D-06.csv',
'75-finger-slow-D-07.csv',
'75-finger-slow-D-08.csv',
'75-finger-slow-D-10.csv',
'77-finger-fast-D-01.csv',
'77-finger-fast-D-02.csv',
'77-finger-fast-D-03.csv',
'77-finger-fast-D-05.csv',
'77-finger-fast-D-06.csv',
'77-finger-fast-D-07.csv',
'77-finger-fast-D-08.csv',
'77-finger-fast-D-09.csv',
'77-finger-fast-D-10.csv',
'77-finger-medium-D-01.csv',
'77-finger-medium-D-02.csv',
'77-finger-medium-D-03.csv',
'77-finger-medium-D-04.csv',
'77-finger-medium-D-05.csv',
'77-finger-medium-D-07.csv',
'77-finger-medium-D-08.csv',
'77-finger-medium-D-09.csv',
'77-finger-medium-D-10.csv',
'77-finger-slow-D-01.csv',
'77-finger-slow-D-02.csv',
'77-finger-slow-D-03.csv',
'77-finger-slow-D-04.csv',
'77-finger-slow-D-05.csv',
'77-finger-slow-D-06.csv',
'77-finger-slow-D-07.csv',
'77-finger-slow-D-09.csv',
'77-finger-slow-D-10.csv',
'85-finger-fast-D-02.csv',
'85-finger-fast-D-04.csv',
'85-finger-fast-D-06.csv',
'85-finger-fast-D-07.csv',
'85-finger-fast-D-09.csv',
'85-finger-fast-D-10.csv',
'85-finger-medium-D-01.csv',
'85-finger-medium-D-02.csv',
'85-finger-medium-D-03.csv',
'85-finger-medium-D-04.csv',
'85-finger-medium-D-05.csv',
'85-finger-medium-D-06.csv',
'85-finger-medium-D-07.csv',
'85-finger-medium-D-08.csv',
'85-finger-medium-D-09.csv',
'85-finger-slow-D-01.csv',
'85-finger-slow-D-02.csv',
'85-finger-slow-D-03.csv',
'85-finger-slow-D-04.csv',
'85-finger-slow-D-05.csv',
'85-finger-slow-D-06.csv',

'85-finger-slow-D-08.csv',
'88-stylus-fast-D-01.csv',
'88-stylus-fast-D-02.csv',
'88-stylus-fast-D-03.csv',
'88-stylus-fast-D-04.csv',
'88-stylus-fast-D-05.csv',
'88-stylus-fast-D-06.csv',
'88-stylus-fast-D-07.csv',
'88-stylus-fast-D-08.csv',
'88-stylus-fast-D-09.csv',
'88-stylus-fast-D-10.csv',
'88-stylus-medium-D-01.csv',
'88-stylus-medium-D-02.csv',
'88-stylus-medium-D-03.csv',
'88-stylus-medium-D-04.csv',
'88-stylus-medium-D-05.csv',
'88-stylus-medium-D-06.csv',
'88-stylus-medium-D-07.csv',
'88-stylus-medium-D-08.csv',
'88-stylus-medium-D-09.csv',
'88-stylus-medium-D-10.csv',
'88-stylus-slow-D-01.csv',
'88-stylus-slow-D-02.csv',
'88-stylus-slow-D-03.csv',
'88-stylus-slow-D-06.csv',
'88-stylus-slow-D-07.csv',
'88-stylus-slow-D-08.csv',
'88-stylus-slow-D-09.csv',
'88-stylus-slow-D-10.csv',
'94-finger-fast-D-01.csv',
'94-finger-fast-D-02.csv',
'94-finger-fast-D-03.csv',
'94-finger-fast-D-04.csv',
'94-finger-fast-D-05.csv',
'94-finger-fast-D-06.csv',
'94-finger-fast-D-07.csv',
'94-finger-fast-D-08.csv',
'94-finger-fast-D-09.csv',
'94-finger-fast-D-10.csv',
'94-finger-medium-D-01.csv',
'94-finger-medium-D-02.csv',
'94-finger-medium-D-05.csv',
'94-finger-medium-D-07.csv',
'94-finger-medium-D-08.csv',
'94-finger-medium-D-09.csv',
'94-finger-medium-D-10.csv',
'94-finger-slow-D-01.csv',
'94-finger-slow-D-02.csv',
'94-finger-slow-D-03.csv',
'94-finger-slow-D-04.csv',
'94-finger-slow-D-05.csv',
'94-finger-slow-D-06.csv',
'94-finger-slow-D-07.csv',
'94-finger-slow-D-08.csv',
'94-finger-slow-D-09.csv',
'94-finger-slow-D-10.csv',
'95-stylus-fast-D-01.csv',
'95-stylus-fast-D-02.csv',
'95-stylus-fast-D-03.csv',
'95-stylus-fast-D-04.csv',
'95-stylus-fast-D-05.csv',
'95-stylus-fast-D-06.csv',
'95-stylus-fast-D-07.csv',
'95-stylus-fast-D-08.csv',
'95-stylus-fast-D-09.csv',
'95-stylus-medium-D-01.csv',
'95-stylus-medium-D-02.csv',
'95-stylus-medium-D-03.csv',
'95-stylus-medium-D-04.csv',
'95-stylus-medium-D-05.csv',
'95-stylus-medium-D-06.csv',
'95-stylus-medium-D-07.csv',
'95-stylus-medium-D-08.csv',
'95-stylus-medium-D-09.csv',
'95-stylus-medium-D-10.csv',
'95-stylus-slow-D-01.csv',
'95-stylus-slow-D-02.csv',
'95-stylus-slow-D-03.csv',
'95-stylus-slow-D-04.csv',
'95-stylus-slow-D-06.csv',
'95-stylus-slow-D-07.csv',
'95-stylus-slow-D-09.csv',
'95-stylus-slow-D-10.csv',
'98-stylus-fast-D-01.csv',
'98-stylus-fast-D-02.csv',
'98-stylus-fast-D-03.csv',
'98-stylus-fast-D-04.csv',
'98-stylus-fast-D-05.csv',
'98-stylus-fast-D-06.csv',
'98-stylus-fast-D-07.csv',
'98-stylus-fast-D-08.csv',
'98-stylus-fast-D-09.csv',
'98-stylus-fast-D-10.csv',
'98-stylus-medium-D-01.csv',
'98-stylus-medium-D-02.csv',
'98-stylus-medium-D-03.csv',
'98-stylus-medium-D-04.csv',
'98-stylus-medium-D-05.csv',
'98-stylus-medium-D-06.csv',
'98-stylus-medium-D-07.csv',
'98-stylus-medium-D-08.csv',
'98-stylus-medium-D-09.csv',
'98-stylus-medium-D-10.csv',
'98-stylus-slow-D-01.csv',
'98-stylus-slow-D-03.csv',
'98-stylus-slow-D-04.csv',
'98-stylus-slow-D-05.csv',
'98-stylus-slow-D-06.csv',
'98-stylus-slow-D-07.csv',
'98-stylus-slow-D-08.csv',

'98-stylus-slow-D-10.csv',
'99-finger-fast-D-01.csv',
'99-finger-fast-D-02.csv',
'99-finger-fast-D-03.csv',
'99-finger-fast-D-04.csv',
'99-finger-fast-D-05.csv',
'99-finger-fast-D-06.csv',
'99-finger-fast-D-07.csv',
'99-finger-fast-D-08.csv',
'99-finger-fast-D-09.csv',
'99-finger-fast-D-10.csv',
'99-finger-medium-D-01.csv',
'99-finger-medium-D-02.csv',
'99-finger-medium-D-03.csv',
'99-finger-medium-D-04.csv',
'99-finger-medium-D-05.csv',
'99-finger-medium-D-06.csv',
'99-finger-medium-D-07.csv',
'99-finger-medium-D-08.csv',
'99-finger-medium-D-09.csv',
'99-finger-medium-D-10.csv',
'99-finger-slow-D-01.csv',
'99-finger-slow-D-03.csv',
'99-finger-slow-D-04.csv',
'99-finger-slow-D-06.csv',
'99-finger-slow-D-07.csv',
'99-finger-slow-D-08.csv',
'99-finger-slow-D-09.csv',
    ];

    for sampleName in sampleNames1:
        dataset.plot(sampleName=sampleName)


if mode == 26:
    d = NormalDistribution(2.0, 0.5)
    samples = [d.sample() for i in range(10000)]
    plt.hist(samples, edgecolor='c', color='c', bins=50)
    plt.show()