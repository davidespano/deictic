# Libraries
from gesture.modellingGesture import Parse
from model import *
from dataset import *
from gesture import *
from test import *

class Joint:

    def __init__(self, x, y, z, nome = None):
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

    for index_frame in range(0, n_frame):
        i = index_frame*66
        frame = Frame(index_frame)

        for j in range(0, 66, 3):
            x = float(f[i+j+0])
            y = float(f[i+j+1])
            z = float(f[i+j+2])
            joint = Joint(x,y,z)
            frame.joints.append(joint)

        frames.append(frame)

    return frames















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



