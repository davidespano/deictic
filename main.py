from dataset import *
from gesture import *
from test import *



# Main
baseDir = '/home/alessandro/PycharmProjects/deictic/repository/'
n_states = 6 # Numero stati
n_samples = 20
iterations = 10 # k-fold cross-validation
mode = 6


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
        #folders = ['caret', 'check', 'delete_mark', 'left_sq_bracket', 'right_sq_bracket',
        #           'star', 'triangle', 'v', 'x']
        folders = ['v', 'caret', 'left_sq_bracket', 'right_sq_bracket',
                              'x', 'delete_mark', 'triangle', 'rectangle']
        gestureDir = baseDir + 'deictic/unica-dataset/resampled/'
        type = 'unica-'
    # 1Dollar
    elif mode == 0:
        #folders = ['arrow', 'caret', 'check', 'circle', 'delete_mark', 'left_curly_brace', 'left_sq_bracket',
        #           'pigtail', 'question_mark', 'rectangle', 'right_curly_brace', 'right_sq_bracket',
        #           'star', 'triangle', 'v', 'x']
        folders = [ 'triangle', 'x', 'rectangle', 'circle', 'check', 'caret', 'question_mark', 'arrow',
                    'left_sq_bracket', 'right_sq_bracket', 'v', 'delete_mark', 'left_curly_brace', 'right_curly_brace',
                    'star', 'pigtail']

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
        folders = ['triangle', 'x', 'rectangle', 'circle', 'check', 'caret', 'question_mark', 'arrow',
                   'left_sq_bracket', 'right_sq_bracket', 'v', 'delete_mark', 'left_curly_brace', 'right_curly_brace',
                   'star', 'pigtail']
        gestureDir = baseDir + 'deictic/1dollar-dataset/resampled/'
        list_filesDir = baseDir + 'deictic/1dollar-dataset/ten-cross-validation/'
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
if mode in [6, 7, 8, 9, 10]:
    ## Adhoc hmm - 1Dollar
    if mode == 6:
        list_gesture = [
            ("triangle", n_states * 3),
            ("x", n_states * 3),
            ("rectangle", n_states * 4),
            ("circle", n_states * 4),
            ("check", n_states * 2),
            ("caret", n_states * 2),
            ("question_mark", n_states * 4),
            ("arrow", n_states * 4),
            ("left_sq_bracket", n_states * 3),
            ("right_sq_bracket", n_states * 3),
            ("v", n_states*2),
            ("delete_mark", n_states*4),
            ("left_curly_brace", n_states * 6),
            ("right_curly_brace", n_states * 6),
            ("star", n_states*4),
            ("pigtail", n_states*4) ]
        #list_gesture = [("caret", n_states*2), ("v", n_states*2)]
        gestureDir = baseDir + 'deictic/1dollar-dataset/resampled/'
        list_filesDir = baseDir + 'deictic/1dollar-dataset/ten-cross-validation/'
        n_features = 2

        gestures = [i[0] for i in list_gesture]
        results = None

        # Create hmm gesture, training and testing sequences
        hmms = []
        confusion = numpy.zeros((len(list_gesture), len(list_gesture)))
        for k in range(0, iterations):
            hmms = []
            for gesture in list_gesture:
                # Create and training hmm
                print("---------------  gesture: {0}  -----------------".format(gesture[0]));
                # Training dataset
                training_dataset = CsvDataset(gestureDir + gesture[0] + '/'). \
                    read_ten_cross_validation_dataset(list_filesDir + gesture[0] + '/', type='train')
                # Create and training hmm
                hmms.append(create_hmm_gesture(gesture[0], training_dataset, gesture[1], n_features))

            t = test(hmms, gestureDir, gestures, plot=False)
            t.ten_cross_validation(list_filesDir, k)
            print("K = {}".format(k))
            print(t.results)
            confusion = confusion + t.results;
        print("--------------------- finale ------------------------")
        print(confusion)

    ## Adhoc hmm - MDollar
    if mode == 7:
        list_gesture = [
            ("T", 2),
            ("N", 3),
            ("D", 2),
            ("P", 2),
            ("X", 2),
            ("H", 3),
            ("I", 3),
            ("exclamation_point", 2),
            ("null", 2),
            ("arrowhead", 2),
            ("pitchfork", 2),
            ("six_point_star", 2),
            ("asterisk", 3),
            ("half_note", 3)
            ]
        #list_gesture = [("D", 2), ("X", 2)]

        list_avoid = {
            "D" : [0,3], "H" : [], "I": [], "N" : [], "P": [0,3], "T": [1,2,3], "X": [], "arrowhead": [1], "asterisk" : [],
            "exclamation_point": [], "half_note": [0],
            "null": [], "pitchfork":[],
            "six_point_star": []
        }

        gestureDir = baseDir + 'deictic/mdollar-dataset/resampled/'
        list_filesDir = baseDir + 'deictic/mdollar-dataset/ten-cross-validation/'
        n_features = 2

        # Training
        gestures = dict()

        complete = numpy.zeros((len(list_gesture), len(list_gesture)))

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
                    file = file.split('#')
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
                                weights = [1,1, 100000],
                                stroke = gesture[1])
                            )
                gestures[gesture[0]] = hmms

            print("inizio il test")
            results = compares_deictic_models(gestures, gestureDir, ten_fold=True)
            complete = complete + results
            print("K = {}".format(k))
            print(results)

        print('============== complete ================')
        print(complete)

    if mode == 9:
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

    if mode == 10:
        list_gesture = [('v', n_states * 2), ('caret', n_states * 2), ('left_sq_bracket', n_states * 3),
                        ('right_sq_bracket',n_states * 3), ('x',n_states * 3), ('delete_mark', n_states * 3),
                        ('triangle', n_states * 3), ('rectangle', n_states * 3)]

        #list_gesture = [("caret", n_states * 2), ("v", n_states * 2)]
        gestureDir = baseDir + 'deictic/unica-dataset/resampled/'
        list_filesDir = baseDir + 'deictic/unica-dataset/ten-cross-validation/'
        n_features = 2

        gestures = [i[0] for i in list_gesture]
        results = None

        # Create hmm gesture, training and testing sequences


        confusion = numpy.zeros((len(list_gesture), len(list_gesture)))



        for k in range(0, iterations):
            hmms = []
            for gesture in list_gesture:
                # Training dataset
                training_dataset = CsvDataset(gestureDir + gesture[0] + '/'). \
                    read_ten_cross_validation_dataset(list_filesDir + gesture[0] + '/', type='train')
                # Create and training hmm
                hmms.append(create_hmm_gesture(gesture[0], training_dataset, gesture[1], n_features))

            t = test(hmms, gestureDir, gestures, plot=False)
            t.ten_cross_validation(list_filesDir, k)
            print("K = {}".format(k))
            print(t.results)
            confusion = confusion + t.results;
        print("--------------------- finale ------------------------")
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
    dataset = CsvDataset(testDir + "exclamation_point/")
    #dataset.plot()
    dataset.plot(singleMode=True)
    #dataset.plot(sampleName="10-stylus-fast-pitchfork-10.csv")

if mode == 26:
    d = NormalDistribution(2.0, 0.5)
    samples = [d.sample() for i in range(10000)]
    plt.hist(samples, edgecolor='c', color='c', bins=50)
    plt.show()