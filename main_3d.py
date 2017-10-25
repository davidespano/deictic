# Libraries
from gesture.modellingGesture import Parse
from model import *
from dataset import *
from gesture import *
from test import *

class Joint:

    def __init__(self, x, y, z, nome):
        if (isinstance(x, float)):
            self.x = x
        elif (isinstance(y, float)):
            self.y = y
        elif (isinstance(z, float)):
            self.z = z
        else:
            print("Errore, non è un float")
        self.nome = nome

class Frame:

    def __init__(self, num):
        self.joints = []
        self.num = num

    def recupera_joint(self, n):
        for joints in range(0, len(joints)):
            joint_recuperato = joints[n]
            return joint_recuperato



def txt_to_csv():
    file = open("/home/sara/Scrivania/simple-shrec/gesture_1/gesture_1/finger_1/subject_1/essai_1/skeletons_world.txt", "r").read()
    f = file.replace('\n', " ").split(" ")

    n_frame = int(len(f) / 66)
    frames = []
    index_frame = 0

    print(f[0] + " " + f[1] + " " + f[2])# Frame numero 1
    print(f[1320] + " " + f[1321] + " " + f[1322]) #Frame numero 20
    print(f[3300] + " " + f[3301] + " " + f[3302]) #Frame numero 50
    print(f[5940] + " " + f[5941] + " " + f[5942]) #Frame numero 90

    names = ['Wrist', 'Palm', 'thumb_base', 'thumb_first_joint', 'thumb_second_joint', 'thumb_tip', 'index_base',
            'index_first_joint', 'index_second_joint', 'index_tip', 'middle_base', 'middle_first_joint',
            'middle_second_joint', 'middle_tip', 'ring_base', 'ring_first_joint', 'ring_second_joint',
            'ring_tip', 'pinky_base', 'pinky_first_joint', 'pinky_second_joint', 'pinky_tip']

    for index_frame in range(0, n_frame):
        i = index_frame*66
        frame = Frame(index_frame)

        for j in range(0, 66, 3):
            k = 0
            x = float(f[i+j+0])
            y = float(f[i+j+1])
            z = float(f[i+j+2])
            joint = Joint(x,y,z,str(names[k]))
            frame.joints.append(joint)

        frames.append(frame)
        k = k+1
    return frames


    for frames in range(0, len(frames)):
        for frame in range(0, len(frame)):
            palm_joint = joints[1]
            index_tip = joints[9]
    return palm_joint, index_tip



    frames_da_salvare = []















# Indica quale tipo di debug voglio avviare (0 = debug GestureModel, 1 = debug DeclarativeModel, 2 = debug Dataset Shrek in 2 Dimensioni)
debug_mode = 3

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
 txt_to_csv()



