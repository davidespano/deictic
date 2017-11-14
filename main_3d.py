# Libraries
from config import Config
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
# Type
type = "palm"
# type = "index_tip"
print(type)
# Indica quale tipo di debug voglio avviare (0 = debug GestureModel, 1 = debug DeclarativeModel, 2 = debug Dataset Shrek in 2 Dimensioni)
debug_mode = 6

#### Debug GestureModel (Line3D e Point3D) ####
if debug_mode == 0:
    l = Line3D(20,1,5)
    print(l.__str__())
    l.plot()

#### Debug DeclarativeModel (Line3D e Point3D) ####
if debug_mode == 1:
    # todo
    print()

# Convert shrec dataset files from txt to csv
if debug_mode == 3:
    # Selected axis (x=0, y=1, z=2)
    x = 0; y = 1; z = 2
    # Gesture 2, 7, 8, 9, 10, 11, 12, 13
    gestures = [[2,[z,y]], [7,[x,z]], [8,[x,z]], [9,[z,y]], [10,[z,y]], [11,[x,y]], [12,[x,y]], [13,[x,y]]]
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

# Resample
if debug_mode == 4:
    # Gesture 2 (tap), 7 (right), 8(left), 9(up), 10(down), 11(x), 12(+), 13(v) - num_primitives - true/false axisMode
    gestures = [[2, 2, False], [7, 1, False], [8, 1, False], [9, 1, False], [10, 1, False], [11, 3, True], [12, 3, True], [13, 2, True]]

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
        transform1 = NormaliseLengthTransform(axisMode=gesture[2])
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
    gesture = "13"

    dir = "/home/"+person+"/PycharmProjects/deictic/repository/deictic/shrec-dataset/resampled/"+tipo_joint+"/gesture_"+gesture+"/"
    dataset = CsvDataset(dir)
    # Hmm
    parse = Parse(n_states=6, n_samples=20)
    model = parse.parseExpression("shrec-gesture_"+gesture)
    # Visualizza nella stessa immagine: movimento utente ed esempio generato dal model (caratteristica delle catene di markov)
    dataset.plot(singleMode=True)



# Test
if debug_mode == 6:
    gesture_models = \
    {

        'gesture_2': [
            Point(0, 0) + Line(-4, 2),
            Point(0, 0) + Line(-4, 2) + Line(4, -2),
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
            Point(0, 4) + Line(4, -4) + Line(-4, 0) + Line(4,4),
            Point(4, 4) + Line(-4, -4) + Line(0, 4) + Line(4,-4)
        ],

        'gesture_12': [
            Point(0,4) + Line(0,-4) + Line(-2,2) + Line(4,0),
            Point(0,2) + Line(0,2) + Line(0,-4) + Line(-2,2) + Line(4,0),
        ],

        'gesture_13': [
            Point(2, 4) + Line(-2, -4) + Line(-2, 4),
            Point(-2, 4) + Line(2, -4) + Line(2, 4)
        ]
    }
    n_states = 12
    n_samples = 40
    hmms = dict()
    factory = ClassifierFactory()
    factory.setLineSamplesPath(Config.trainingDir)
    factory.setClockwiseArcSamplesPath(Config.arcClockWiseDir)
    factory.setCounterClockwiseArcSamplesPath(Config.arcCounterClockWiseDir)
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
