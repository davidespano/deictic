# Libraries
from gesture.modellingGesture import Parse
from model import *
from dataset import *
from gesture import *
from test import *

class Joint:

    def __init__(self, x, y, z):
        if (isinstance(x, float)):
            self.x = x
        else:
            print("Errore, non è un float")
        if (isinstance(y, float)):
            self.y = y
        else:
            print("Errore, non è un float")
        if (isinstance(z, float)):
            self.z = z
        else:
            print("Errore, non è un float")

    def return_coordinates(self):
        return [self.x,self.y,self.z]

class Frame:

    def __init__(self, num):
        self.joints = []
        self.num = num

    def recupera_joint(self, n):
        joint_recuperato = self.joints[n]
        return joint_recuperato



def txt_to_csv(file_scelto, joint_scelto):
    file = open(file_scelto, "r").read()
    #file = open("/home/ale/Scaricati/HandGestureDataset_SHREC2017/gesture_1/finger_1/subject_1/essai_1/skeletons_world.txt",
                #"r").read()
    f = file.replace('\n', " ").split(" ")

    n_frame = int(len(f) / 66)
    frames = []
    index_frame = 0

    #print(f[0] + " " + f[1] + " " + f[2])# Frame numero 1
    #print(f[1320] + " " + f[1321] + " " + f[1322]) #Frame numero 20
    #print(f[3300] + " " + f[3301] + " " + f[3302]) #Frame numero 50
    #print(f[5940] + " " + f[5941] + " " + f[5942]) #Frame numero 90

    #names = ['Wrist', 'Palm', 'thumb_base', 'thumb_first_joint', 'thumb_second_joint', 'thumb_tip', 'index_base',
            #'index_first_joint', 'index_second_joint', 'index_tip', 'middle_base', 'middle_first_joint',
            #'middle_second_joint', 'middle_tip', 'ring_base', 'ring_first_joint', 'ring_second_joint',
            #'ring_tip', 'pinky_base', 'pinky_first_joint', 'pinky_second_joint', 'pinky_tip']

    # Estrai i frame con tutti i joint dal file
    for index_frame in range(0, n_frame):
        i = index_frame*66
        frame = Frame(index_frame)
        # Estrai tutti i joint relativi al frame in questione
        for j in range(0, 66, 3):
            x = float(f[i+j+0])
            y = float(f[i+j+1])
            z = float(f[i+j+2])
            joint = Joint(x,y,z)
            frame.joints.append(joint)
        # Inserisci il frame nella lista frames
        frames.append(frame)

    # Crea e restituisci due liste che contengono, rispettivamente, tutte le coordinate registrate nei vari frame e relativi
    # a un determinato Joint (nel nostro caso palm (1) e l'index_tip (9)).
    palm_joints = []
    index_tip_joints = []
    for frame in frames:
        palm_joints.append(frame.joints[1].return_coordinates())
        index_tip_joints.append(frame.joints[9].return_coordinates())
    if joint_scelto == 1:
        return palm_joints
    elif joint_scelto == 9:
        return index_tip_joints














# Indica quale tipo di debug voglio avviare (0 = debug GestureModel, 1 = debug DeclarativeModel, 2 = debug Dataset Shrek in 2 Dimensioni)
debug_mode = 4

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


if debug_mode == 3:

    gesture = 14
    finger = 2

    dir = '/home/sara/Scaricati/HandGestureDataset_SHREC2017/gesture_'+str(gesture)+'/finger_'+str(finger)+'/'
    lista_subject = os.listdir(dir)
    for subject in lista_subject:
        path = subject
        lista_essai = os.listdir(dir + subject)

        for essai in lista_essai:
            if finger == 1:
                joint_scelto = 9
                nome_joint = "index_tip_joint"
            else:
                joint_scelto  = 1
                nome_joint = "palm_joint"
            file_da_leggere = dir + subject + '/'+ essai + '/skeletons_world.txt'
            joints = txt_to_csv(file_da_leggere, joint_scelto)
            numpy.savetxt("/home/sara/PycharmProjects/deictic/repository/deictic/shrec-dataset/raw/gesture_"+str(gesture)+"/" + subject + "_" + essai + "_" + nome_joint +".csv", joints, delimiter=',', fmt='%f')


if debug_mode == 4:
    person = "ale"
    #person = "sara"
    gesture = "2"

    dir = "/home/"+person+"/PycharmProjects/deictic/repository/deictic/shrec-dataset/raw/gesture_"+gesture+"/"
    dataset = CsvDataset(dir)
    # Hmm
    parse = Parse(n_states=6, n_samples=20)
    model = parse.parseExpression("shrec-gesture_"+gesture)
    # Visualizza nella stessa immagine: movimento utente ed esempio generato dal model (caratteristica delle catene di markov)
    dataset.plot(singleMode=True, model=model)


if debug_mode == 5:
    # num_primitive + il numero di primitive che costituiscono la gesture (tipo 1 oppure 2 o 3 e così via, a seconda della gesture).
    # Se non ti ricordi quali sono le primitive a disposizione, ricontrolla gli articoli che ti avevo passato.
    num_primitive = 4
    #person = "ale"
    person = "sara"
    gesture = "14"

    inputDir = "/home/"+person+"/PycharmProjects/deictic/repository/deictic/shrec-dataset/raw/gesture_"+gesture+"/"
    outputDir = "/home/"+person+"/PycharmProjects/deictic/repository/deictic/shrec-dataset/resampled/gesture_"+gesture+"/"
    #### Questo codice ti serve per creare le sequenze campionate e normalizzate dei file che hai convertito con il metodo txt_to_csv.  ####
    dataset = CsvDataset(inputDir)
    # Transform
    transform1 = NormaliseLengthTransform(axisMode=True)
    transform2 = ScaleDatasetTransform(scale=100)
    transform3 = CenteringTransform()
    transform5 = ResampleInSpaceTransform(samples=num_primitive*20)
    # Apply transforms
    dataset.addTransform(transform1)
    dataset.addTransform(transform2)
    dataset.addTransform(transform3)
    dataset.addTransform(transform5)
    dataset.applyTransforms(outputDir)

if debug_mode==6:
    # Creazione HMM #
    # Aggiungi le altre gesture rispettando questa nomenclatura (shrec + gesture_x).
    # Questa stringa serve alla funzione che si occupa di creare le hmm:
    # - shrec indica alla classe Parsing che deve prendere la definizione della gesture dalla classe che che hai creato in modelling_gesture
    # - come ti puoi immaginare, gesture_x indica l'espressione di tale gesture presente nella classe che definisce shrec
    lista_gesture =["shrec-gesture_14"]
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
        model = parse.parseExpression(gesture_name)
        # assegna nome all'hmm
        model.name = gesture_name
        # Adds hmm in the list
        hmms.append(model)


