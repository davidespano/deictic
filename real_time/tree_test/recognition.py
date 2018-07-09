from gesture import *
#from test.test import Test
import matplotlib.pyplot as mp
from sys import stdout
import csv
from config import Config

def recognizeByProbability(gesture, gesture_hmms, enable_show):
    # gesture= gesture da riconoscere
    # gesture_hmms= il dizionario contenente gli hmms
    # enable_show=bool per attivare o meno la visualizzazione grafica del riconoscimento

    results = []

    unistroke_mode = True

    path0 = Config.baseDir + 'Tree_test/' + gesture[0] + '/'

    dataset = CsvDataset(path0, type=float)

    # Transform
    transform1 = NormaliseLengthTransform(axisMode=True)
    transform2 = ScaleDatasetTransform(scale=100)
    transform3 = CenteringTransform()
    if unistroke_mode:
        transform5 = ResampleInSpaceTransform(samples=gesture[1])
    else:
        transform5 = ResampleInSpaceTransformMultiStroke(samples=gesture[1], strokes=gesture[2])
    # Apply transforms
    dataset.addTransform(transform1)
    dataset.addTransform(transform2)
    dataset.addTransform(transform3)
    dataset.addTransform(transform5)

    list_of_points = dataset.applyTransforms()

    # Confronto una coppia di punti del dataset con gli hmm dell'albero con la compare, che mi restituisce
    # il nome dell'hmm con la probabilità massima tra i vari hmms
    list_of_prob_fullGesture = []
    someFrames = []
    list_of_prob_someFrames = []

    i = 0  # Counter dei frames

    # Coordinate
    x = 0
    y = 0
    stop_counter = 0  # counter del numero di file letti (dello stesso tipo)
    stop = 0  # counter del numero totale dei file letti
    plot_frames = []  # lista dei frames
    label = ""  # gesture riconosciuta
    prev_type = ""  # precedente tipo di file letto
    prev_filename = ""  # file letto nel precedente ciclo for (per stampare correttamente il titolo del grafico)

    prev_frame = []
    frame = []
    frames_changes = []

    results_list = []  # risultati del riconoscimento realtime

    for ndarray, name in list_of_points:

        # Riconoscimento singolo file (ndarray)
        list_of_prob_fullGesture.append(Test.compare(sequence=ndarray, gesture_hmms=gesture_hmms,
                                                     return_log_probabilities=False))

        # Stampo il grafico finale
        if (someFrames != [] and plot_frames != []):
            mp.title(label)
            mp.suptitle(prev_filename)
            label_changes -= 1

            for x, y in plot_frames:
                if (label_changes == 0):
                    mp.plot(x, y, 'bo')
                if (label_changes == 1):
                    mp.plot(x, y, 'ro')
                if (label_changes == 2):
                    mp.plot(x, y, 'go')
                if (label_changes == 3):
                    mp.plot(x, y, 'yo')
                if (label_changes == 4):
                    mp.plot(x, y, 'ro')
                if (label_changes == 5):
                    mp.plot(x, y, 'go')
                if (label_changes == 6):
                    mp.plot(x, y, 'bo')
                if (label_changes == 7):
                    mp.plot(x, y, 'ro')
                if (label_changes == 8):
                    mp.plot(x, y, 'yo')

            if (enable_show):
                # print(" Frames_changes: " + frames_changes.__str__())
                mp.show()
                frames_changes = []
                frame = []
                prev_frame = []

            results_list.append(prob)

        i = 0  # Inizializzo il counter dei frame
        someFrames = []  # Inizializzo la lista contenente i frame quando cambio file

        label_changes = 0
        plot_frames = []

        # Dopo 12 file, interrompo il ciclo for
        if (stop == 12):
            break

        # Per far si che legga solo 4 file fast/medium/slow, devo usare un counter, che si incrementa di 1 ogni volta
        # che incontra un file dello stesso tipo, ma che si resetta se quello precedente era di un altro tipo
        # Se prev_type è una stringa vuota, setto stop_counter a -1 perchè si tratta della prima iterazione
        # (Altrimenti la prima tipologia di file verrebbe letta solo 3 volte, in base al mio algoritmo)

        if (prev_type == ""):
            stop_counter = -1

        if ("fast" in name):
            if (prev_type == "medium"):
                stop_counter = 0
                prev_type = "fast"
            elif (prev_type == "slow"):
                stop_counter = 0
                prev_type = "fast"
            else:
                prev_type = "fast"
                stop_counter += 1
        if ("medium" in name):
            if (prev_type == "fast"):
                stop_counter = 0
                prev_type = "medium"
            elif (prev_type == "slow"):
                stop_counter = 0
                prev_type = "medium"
            else:
                prev_type = "medium"
                stop_counter += 1
        if ("slow" in name):
            if (prev_type == "medium"):
                stop_counter = 0
                prev_type = "slow"
            elif (prev_type == "fast"):
                stop_counter = 0
                prev_type = "slow"
            else:
                prev_type = "slow"
                stop_counter += 1
        if (stop_counter < 4):
            stop += 1


            for ndarray1 in ndarray:
                i += 1

                # Riconoscimento attraverso un array di frames (viene incrementato di un frame per ogni ciclo)
                someFrames.append(ndarray1)
                plot_frames.append((ndarray1[0], ndarray1[1]))

                frame = ndarray1

                prev_label = label
                label, prob = Test.compare(sequence=someFrames, gesture_hmms=gesture_hmms,
                                           return_log_probabilities=True)

                # Stampa del numero del frame sul frame stesso
                # mp.text(ndarray1[0], ndarray1[1], i)

                # Stampo grafico parziale
                a = (prev_label == label)
                if ((a == False) and label != None):
                    for x, y in plot_frames:
                        if (label_changes == 0):
                            mp.plot(x, y, 'bo')
                        if (label_changes == 1):
                            mp.plot(x, y, 'ro')
                        if (label_changes == 2):
                            mp.plot(x, y, 'go')
                        if (label_changes == 3):
                            mp.plot(x, y, 'yo')
                        if (label_changes == 4):
                            mp.plot(x, y, 'ro')
                        if (label_changes == 5):
                            mp.plot(x, y, 'go')
                        if (label_changes == 6):
                            mp.plot(x, y, 'bo')
                    label_changes += 1

                    x, y = plot_frames[int(len(plot_frames) / 2)]
                    mp.text(x, y, label)
                    plot_frames = []

                # Stampa di aggiornamento (stampo in questo modo per sostituire sempre la riga attuale)
                stdout.write("\rCalcolo probabilità FILE: " + name + "  Frame " + i.__str__() +
                             "  Gesture riconosciuta: " + label.__str__())
                stdout.flush()
                prev_filename = name
                # time.sleep(0.025)

                y += 1
                list_of_prob_someFrames.append(label)

    i = 0

    # Result full è un dizionario gesture -> numero di riconoscimenti....full perchè appartiene al fullgesture
    result_full = {i: list_of_prob_fullGesture.count(i) for i in list_of_prob_fullGesture}
    print("\nLa gesture con il maggior numero di match nella modalità fullgesture è: ")
    max = -1

    # Cerco il massimo valore nel dizionario
    for label in list(result_full.keys()):
        if (max == -1):
            max = (label, (result_full[label]))
        if (result_full[label] > max[1]):
            max = (label, (result_full[label]))

    # Lo stampo
    print(max)
    print("Risultati complessivi: " + result_full.__str__())

    print("\nRisultati modalità someframes: ")
    print(results_list)

# Funzione extra (Riconoscimento di una gesture tramite confronto)
def recognizeByCompare(tree, gesture):
    # Se il nodo ha figli, li ''visito'', altrimenti stampo che non ne ha
    if (tree.children != []):
        for figlio in tree.children:
            temp = (list(gesture.keys()))[0]
            temp = gesture[temp][0]
            if (figlio.gestures != None):
                gesture_figlio = [eval(figlio.gestures)][0]
                if (temp.__str__() == gesture_figlio.__str__()):
                    print("Gesture riconosciuta come " + figlio.name)
        for figlio in tree.children:
            figlio.recognizeByCompare(figlio, gesture)

#Funzione di debug per aiutare la scrittura a mano dei frames nel file csv
def plotCsvFile(gesture, path0=-1):
    if (path0 == -1): path0 = Config.baseDir + 'Tree_test/' + gesture[0] + '/'

    dataset = CsvDataset(path0, type=float)

    dataset_iter=dataset.getDatasetIterator()

    list=[]
    for file in dataset_iter:
        i=0

        for ndarray in dataset.readFile(file):
            i+=1
            mp.plot(ndarray[0], ndarray[1], 'bo')
            mp.text(ndarray[0], ndarray[1], i)

        mp.title(file.__str__())
        mp.show()

# todo: add to tree?
def readChangePrimitivesFile(path):
    path0=path
    #path0 = Config.baseDir+ 'Tree_test/manualRecognition/changePrimitives.csv'

    f = open(path0, 'rt')
    try:
        changePrimitivesDict={}
        reader = csv.reader(f)
        i=1
        for row in reader:
            if(i%13!=1):
                row=re.split(r"[;]", row[0])
                file_name=row[0]
                part_list=[]
                for part in row:
                    if(part is not file_name and part is not ""):
                        try:
                            temp=((re.split(r"[:]", part))[1])

                            temp=((re.findall(r'\d+', temp))[0])

                        except IndexError:
                            print("ERRORE nella suddivisione di pr_x dal numero (es pr_3 da 11): " + (re.split(r"[:]", part)).__str__())
                            print("Non è stato inserito un numero (ma un altro carattere) in una delle pr_x della riga: " + row.__str__())
                            print("Oppure è stato inserito un carattere (diverso dallo spazio) dopo l'ultima pr_x: X (ad esempio pr_5: 45;)\n")

                        temp=float(temp)
                        temp=int(temp)
                        part_list.append(temp)

                changePrimitivesDict.setdefault(file_name, part_list)
            i+=1
        return changePrimitivesDict
    finally:
        f.close()
