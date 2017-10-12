# Libraries
from model import *
from dataset import *
from gesture import *
from test import *
# Plot
import matplotlib.pyplot as plt


# Indica quale tipo di debug voglio avviare (0 = debug GestureModel, 1 = debug DeclarativeModel, 2 = debug Dataset Shrek in 2 Dimensioni)
debug_mode = 2

#### Debug GestureModel (Line3D e Point3D) ####
if debug_mode == 0:
    l = Line3D(0,0,0)
    print(l.__str__())
    #l.plot()

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
        # Visualizzo questa sequenza di punti usando la libreri matplotlib
        result = numpy.array(sequence).astype('float')
        plt.axis("equal")
        plt.plot(result[:, 0], result[:, 1])
        plt.title(gesture)
        plt.show()