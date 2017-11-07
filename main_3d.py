# Libraries
from gesture.modellingGesture import Parse
from model import *
from dataset import *
from gesture import *
from test import *
import matplotlib.pyplot as plt
import os

class Joint:

    def __init__(self, values = []):
        if not isinstance(values, list):
            raise ("axis must be a list!")
        self.values = values;

    def return_coordinates(self):
        return self.values
        #return [self.x,self.y,self.z]

class Frame:

    def __init__(self, num):
        self.joints = []
        self.num = num

    def recupera_joint(self, n):
        joint_recuperato = self.joints[n]
        return joint_recuperato



def txt_to_csv(file_scelto, joint_scelto, axis = []):
    if not isinstance(axis, list):
        raise("axis must be a list!")

    # Read file
    file = open(file_scelto, "r").read()
    f = file.replace('\n', " ").split(" ")

    n_frame = int(len(f) / 66)
    frames = []

    # Estrai i frame con tutti i joint dal file
    for index_frame in range(0, n_frame):
        i = index_frame*66
        frame = Frame(index_frame)
        # Estrai tutti i joint relativi al frame in questione
        for j in range(0, 66, 3):
            # Get coordinates based on axis values
            values = []
            for item in axis:
                if item == 0:
                    values.append(float(f[i+j+0]))
                elif item == 1:
                    values.append(float(f[i+j+1]))
                elif item == 2:
                    values.append(float(f[i+j+2]))
            # Create joint
            joint = Joint(values)
            # Add joint to frame
            frame.joints.append(joint)
        # Inserisci il frame nella lista frames
        frames.append(frame)

    # Crea e restituisci due liste che contengono, rispettivamente, tutte le coordinate registrate nei vari frame e relativi
    # a un determinato Joint (nel nostro caso palm (1) e l'index_tip (9)).
    joints = []
    for frame in frames:
        joints.append(frame.joints[joint_scelto].return_coordinates())
    return joints










# Person
person = "ale"
#person = "sara"
# Indica quale tipo di debug voglio avviare (0 = debug GestureModel, 1 = debug DeclarativeModel, 2 = debug Dataset Shrek in 2 Dimensioni)
debug_mode = 7

#### Debug GestureModel (Line3D e Point3D) ####
if debug_mode == 0:
    l = Line3D(20,1,5)
    print(l.__str__())
    l.plot()

#### Debug DeclarativeModel (Line3D e Point3D) ####
if debug_mode == 1:
    # todo
    print()

#### Debug Dataset Shrek in 2 dimensioni ####
if debug_mode == 2:
    # Gesture dataset
    gestures = ['shrec-grab', 'shrec-tap', 'shrec-expand', 'shrec-pinch', 'shrec-rotation_clockwise',
                'shrec-rotation_counter_clockwise', 'shrec-swipe_right', 'shrec-swipe_left',
                'shrec-swipe_up', 'shrec-swipe_down', 'shrec-swipe_x', 'shrec-swipe_plus',
                'shrec-swipe_v', 'shrec-shake']

    # Creazione hmm
    n_states = 6
    n_samples = 20
    hmms = []
    parse = Parse(n_states, n_samples)
    for gesture in gestures:
        # 'Per quale gesture sta creando l'hmm?'
        print('Creazione hmm per la gesture: '+gesture)
        # Creo l'hmm (Ricordati di aggiungere nel metodo getModel della classe Shrek di modellingGesture la definizione per tutte le gesture che vuoi definire).
        model = parse.parseExpression(gesture)
        # Gli assegno il nome della gesture a cui fa riferimento
        model.name = gesture
        # Aggiungo l'hmm nella mia lista hmms
        hmms.append(model)

    # Creo e visualizzo un sample generato dal hmm
    for hmm in hmms:
        # Genero una sequenza di punti di esempio
        sequence = hmm.sample()
        # Visualizzo questa sequenza di punti usando la libreria matplotlib
        result = numpy.array(sequence).astype('float')
        plt.axis("equal")
        plt.plot(result[:, 0], result[:, 1])
        plt.title(gesture)
        plt.show()


# Convert shrec dataset files from txt to csv
if debug_mode == 3:
    # Selected axis (x=0, y=1, z=2)
    x = 0; y = 1; z = 2
    # Gesture 2, 7, 8, 9, 10, 11, 12, 13
    gestures = [[2,[z,y]], [7,[x,z]], [8,[x,z]], [9,[z,y]], [10,[z,y]], [11,[x,y]], [12,[x,y]], [13,[x,y]]]
    # Type
    #type = "palm"
    type = "index_tip"
    # Finger joint_scelto and nome_joint
    if type == "palm":
        finger = "2"
        joint_scelto = 1
        nome_joint = "palm_joint"
    else:
        finger = "1"
        joint_scelto = 9
        nome_joint = "index_tip_joint"

    for tuple in gestures:
        # Get data
        gesture = tuple[0]
        axis = tuple[1]
        # Save path
        save_dir = "/home/"+person+"/PycharmProjects/deictic/repository/deictic/shrec-dataset/raw/"+type+"/gesture_"+str(gesture)+"/"
        # Create folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dir = '/home/'+person+'/Scaricati/HandGestureDataset_SHREC2017/gesture_'+str(gesture)+'/finger_'+finger+'/'
        lista_subject = os.listdir(dir)
        for subject in lista_subject:
            path = subject
            lista_essai = os.listdir(dir + subject)
            for essai in lista_essai:
                file_da_leggere = dir + subject + '/'+ essai + '/skeletons_world.txt'
                joints = txt_to_csv(file_da_leggere, joint_scelto, axis)
                numpy.savetxt(save_dir + "gesture_"+str(gesture) +"_"+ subject +"_"+ essai +"_"+ nome_joint +".csv", joints, delimiter=',', fmt='%f')
        print("Gesture "+str(gesture)+" converted.")
# Plotting raw data
if debug_mode == -3:
    # Gesture 2, 7, 8, 9, 10, 11, 12, 13
    gesture = 9
    type = "palm"
    #type = "index_tip"

    dir_dataset = "/home/"+person+"/PycharmProjects/deictic/repository/deictic/shrec-dataset/raw/"+type+"/gesture_"+str(gesture)+"/"
    #dir_dataset = "/home/"+person+"/PycharmProjects/deictic/repository/original/shrec-dataset/"+type+"/gesture_"+str(gesture)+"/"
    dataset = CsvDataset(dir_dataset)
    dataset.plot(singleMode=True)



# Resample
if debug_mode == 4:
    # Gesture 2 (tap), 7 (right), 8(left), 9(up), 10(down), 11(x), 12(+), 13(v) - num_primitives
    gestures = [[2, 2], [7, 1], [8, 1], [9, 1], [10, 1]]
    #type = "palm"
    type = "index_tip"

    for gesture in gestures:
        # num_primitive + il numero di primitive che costituiscono la gesture (tipo 1 oppure 2 o 3 e così via, a seconda della gesture).
        # Se non ti ricordi quali sono le primitive a disposizione, ricontrolla gli articoli che ti avevo passato.
        gesture_name = str(gesture[0])
        num_primitive = gesture[1]

        inputDir = "/home/"+person+"/PycharmProjects/deictic/repository/deictic/shrec-dataset/raw/"+type+"/gesture_"+gesture_name+"/"
        outputDir = "/home/"+person+"/PycharmProjects/deictic/repository/deictic/shrec-dataset/resampled/"+type+"/gesture_"+gesture_name+"/"
        # Create folder
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        #### Questo codice ti serve per creare le sequenze campionate e normalizzate dei file che hai convertito con il metodo txt_to_csv.  ####
        dataset = CsvDataset(inputDir)
        # Transform
        transform1 = NormaliseLengthTransform(axisMode=False)
        transform2 = ScaleDatasetTransform(scale=100)
        transform3 = CenteringTransform()
        transform5 = ResampleInSpaceTransform(samples=num_primitive*20)
        # Apply transforms
        dataset.addTransform(transform1)
        dataset.addTransform(transform2)
        dataset.addTransform(transform3)
        dataset.addTransform(transform5)
        dataset.applyTransforms(outputDir)
        print("Gesture " + str(gesture) + " resampled.")

# Debug plot
if debug_mode == 5:
    gesture = "10"
    #tipo_joint = "index_tip"
    tipo_joint = "palm"

    dir = "/home/"+person+"/PycharmProjects/deictic/repository/deictic/shrec-dataset/resampled/"+tipo_joint+"/gesture_"+gesture+"/"
    dataset = CsvDataset(dir)
    # Hmm
    parse = Parse(n_states=6, n_samples=20)
    model = parse.parseExpression("shrec-gesture_"+gesture)
    # Visualizza nella stessa immagine: movimento utente ed esempio generato dal model (caratteristica delle catene di markov)
    dataset.plot(singleMode=True)



# Test
if debug_mode==6:
    # Type
    type = "palm"
    # type = "index_tip"

    # Creazione HMM #
    # Aggiungi le altre gesture, questa stringa serve alla funzione che si occupa di creare le hmm.
    lista_gesture =["gesture_7", "gesture_9"]#, "gesture_11", "gesture_12", "gesture_13", "gesture_14"]

    # Lista che conterrà tutte le hmm indicate in lista_gesture
    hmms = []
    # Creates models #
    # n_states = numero di stati complessivi che costituiscono le hmm delle singole primitive
    # [Quando si fanno dei test questo numero deve essere lo stesso per tutte le gesture da creare e testare]
    n_states = 6
    # n_samples = elemento usato nella fase di training
    # [Quando si fanno dei test questo numero deve essere lo stesso per tutte le gesture da creare e testare]
    n_samples = 20
    parse = Parse(n_states, n_samples)
    for gesture_name in lista_gesture:
        # gesture_def è una tupla #
        # Stampa il nome della gesture che sta gestendo
        print(gesture_name)
        # crea hmm
        if(type == "palm"):
            model = parse.parseExpression("shrec-"+gesture_name)
        else:
            model = parse.parseExpression("shrec2-"+gesture_name)
        # assegna nome all'hmm
        model.name = gesture_name
        # Adds hmm in the list
        hmms.append(model)

    # Test #
    # test è la classe che si occupa di eseguire appunto di eseguire i test su un certo insieme di hmm e di stamparne i risultati.
    # in questo caso siamo interessati alla matrice di confusione, una matrice nxn
    # [dove n è il numero delle gesture, e in riga e in colonna abbiamo una gesture: quindi riga 1 = gesture_2, colonna 1 = gesture_2 e così via]
    # l'obiettivo è duplice:
    # - quantificare i file di un certo dataset che vengono riconosciuti dall'hmm che descrive quel dataset
    # - quantificare gli eventuali errori (e nel caso sapere da quale hmm vengano invece riconosciuti)
    # La situazione ottimale è una matrice diagonale
    datasetDir = "/home/" + person + "/PycharmProjects/deictic/repository/deictic/shrec-dataset/resampled/"+type+"/"
    t = test(hmms, lista_gesture, datasetDir)
    results = t.all_files()

# Test v.2
if debug_mode == 7:
    lista_gesture = ["gesture_2", "gesture_7", "gesture_8", "gesture_9","gesture_10"]  # , "gesture_11", "gesture_12", "gesture_13", "gesture_14"]
    gesture_models = \
    {

        'gesture_2': [
            Point(0, 0) + Line(-4, 4) + Line(0, -4),
            Point(0, 0) + Line(-4, 4) + Line(4, -4),
            #Point(0, 0) + Line(0, -4) + Line(-4, 2),
            #Point(0, 0) + Line(2, -4) + Line(2, 4),
        ],

        'gesture_7': [
            Point(0, 0) + Line(4, 0),
            Point(0,0) + Line(4,0) + Line(-4,0)
        ],

        'gesture_8': [
            Point(4, 0) + Line(-4, 0),
            Point(4, 0) + Line(-4, 0) + Line(4, 0)
        ],

        'gesture_9': [
            Point(0, 0) + Line(0, 4),
            Point(0, 0) + Line(0, 4) + Line(0, -4)
        ],

        'gesture_10': [
            Point(0, 4) + Line(0, -4),
            Point(0, 4) + Line(0, -4) + Line(0, 4)
        ],

        'gesture_11': [
            Point(0, 4) + Line(0, -4),
            Point(0, 4) + Line(0, -4) + Line(0, 4)
        ],

        'gesture_12': [
            Point(0, 4) + Line(4,-4) + Line(-4,0) + Line(4,4),
            Point(4,0) + Line(-4,4) + Line(0,-4) + Line(4,4),
            Point(0,0) + Line(4,4) + Line(0,-4)+Line(-4,4),
            Point(0, 4) + Line(0, -4) + Line(0, 4)
        ],

        'gesture_13': [
            Point(0, 4) + Line(0, -4),
            Point(0, 4) + Line(0, -4) + Line(0, 4)
        ]
    }
    # Type
    type = "palm"
    #type = "index_tip"
    n_states = 6
    n_samples = 20
    hmms = dict()
    factory = ClassifierFactory()
    factory.setLineSamplesPath(trainingDir)
    factory.setClockwiseArcSamplesPath(arcClockWiseDir)
    factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
    factory.states = n_states
    factory.spu = n_samples
    for k in gesture_models.keys():
        print('Group {0}'.format(k))
        hmms[k] = []
        for gesture in gesture_models[k]:
            model, edges = factory.createClassifier(gesture)
            hmms[k].append(model)

    datasetDir = "/home/" + person + "/PycharmProjects/deictic/repository/deictic/shrec-dataset/resampled/"+type+"/"
    t = test_dict(hmms, datasetDir)



# Debug comparing raw and resampled data
if debug_mode == 8:
    # Person
    person = "ale"
    #person = "sara"
    # Type
    type = "palm"
    #type = "index_tip"
    # Gesture
    gesture = 10

    # raw
    dataset_raw = CsvDataset("/home/" + person + "/PycharmProjects/deictic/repository/original/shrec-dataset/"+type+"/gesture_"+str(gesture)+"/")
    # resampled
    dataset_resampled = CsvDataset("/home/" + person + "/PycharmProjects/deictic/repository/deictic/shrec-dataset/resampled/"+type+"/gesture_"+str(gesture)+"/")

    # Plot data
    for file_raw in dataset_raw.readDataset():
        fig, ax = plt.subplots(figsize=(10, 15))
        # Get resampled data
        raw_data = file_raw[0]
        #resampled_data = dataset_resampled.readFile(file_raw[1])
        raw_plot = plt.plot(raw_data[:,0], raw_data[:,1], color="b")
        ax.scatter(raw_data[:,0], raw_data[:,1])
        for i in range(0, len(raw_data)):
            ax.annotate(str(i), (raw_data[i, 0], raw_data[i, 1]))
        #resampled_plot = plt.plot(file_resampled[:,0], file_resampled[:,1])
        # legend
        #plt.legend((raw_plot[0], resampled_plot[0]), ('raw', 'resampled'), loc='lower right')
        plt.title(file_raw[1])
        plt.axis('equal')
        plt.show()
