import re
import more_itertools as mit
from gesture import *
from test.test import Test
import matplotlib.pyplot as mp
from sys import stdout
import time

class Tree(object):

    def __init__(self, name, gestures=None, children=[]):
        if(isinstance(name, str)):
            self.name = name.__str__()
            self.gestures=gestures
            self.children = []

    def add_child(self, node):
        if (isinstance(node, Tree)):
            self.children.append(node)

    # Splitta la esture nei vari rami dell'albero (per adesso gestisce solo le sequenze)
    def gestureFactory(self, dictionary):

        primitives_dictionary = {}
        i=0
        list_of_operators=[] #lista ordinata degli operatori usati nell'espressione
        list_of_expressions=list(dictionary.keys())
        temp=""

        #Scomposizione della gesture
        for gesture_label in dictionary.keys():
            for expression in dictionary[gesture_label]:


                temp=expression.__str__()

                #Trovo la posizione degli operandi
                sequence=list(mit.locate(temp, lambda x: x == "+")) #sequence
                parallel=list(mit.locate(temp, lambda x: x == "*")) # parallel
                choise=list(mit.locate(temp, lambda x: x == "|")) #choise
                #iter=list(mit.locate(temp, lambda x: x == "")) #iteration (non so che simbolo si usi)

                #Inserisco la posizione degli operandi nella lista e la ordino numericamente
                list_of_operators=sequence+parallel+choise#+iter
                list_of_operators.sort()

                #Sostituisco la posizione con l'operando
                for i, operator in enumerate(list_of_operators):
                    for a in sequence:
                        if(operator==a):
                            list_of_operators[i]='+'
                    for b in parallel:
                        if(operator==b):
                            list_of_operators[i]='*'
                    for c in choise:
                        if(operator==c):
                            list_of_operators[i]='|'
                    #for d in piu:
                    #    if(operator==d):
                    #        list_of_operators[i]=''


                #WIP, stavo cercando di generalizzare la scomposizione, ma ancora non funziona
                if (parallel != [] and choise != []):
                    temp=self.splitCompositeGesture(list_of_operators, temp, half = int(len(list_of_operators) / 2))


                i=0

                #Serve per la scomposizione quando ci sono solo sequence
                if (parallel == [] and choise==[]):
                    temp=[temp]

            #Assegno ad ogni etichetta (gesture_label) la sua chiave, cioè la scomposizione della gesture (self.createGestures(element))
            for element in temp:
                primitives_dictionary.setdefault(gesture_label.__str__(), self.createGestures(element))

        return primitives_dictionary, list_of_operators

    #Wip, non ancora completata/usata
    def splitCompositeGesture(self, list_of_operators, temp, half):

        temp = re.split(r'[\|\*]', temp)
        return temp

    def createGestures(self, element):
        i = 0
        primitives_dictionary={}

        #Scompongo la gesture composta nelle primitive (solo sequence)
        element=re.split(r'[\+]', element)

        # Inizializzo il nuovo dizionario
        while (i < len(element)):

            if (i == 0):
                primitives_dictionary.setdefault("Gesture " + "1", [])
                i += 1
            else:
                primitives_dictionary.setdefault("Gesture " + i.__str__(), [])
                i += 1

        i = 0

        #Lista inversa che mi serve per numerare le sottogesture
        inverse_range = list(reversed(range(len(list(primitives_dictionary.keys())))))

        # Inserimento primitive nel nuovo dizionario
        for primitiva in element:
            for y in range(len(list(primitives_dictionary.keys()))):
                if (i < (len(element) - inverse_range[y])):
                    primitives_dictionary["Gesture " + (y + 1).__str__()].append(primitiva)

            i += 1

        return primitives_dictionary

    def createTree(self, dictionary, indice, chiavi, operators):

        # radice
        if (indice == 0):
            radice = Tree(name='Radice')

        for ramo in chiavi:
            indice=0

            temp=Tree(name=ramo)

            radice.add_child(temp)
            dict_ramo=dictionary[ramo]
            chiavi_ramo=list(dict_ramo.keys())
            lunghezza=chiavi_ramo.__len__()

            while(indice<lunghezza):


                #if (operators[indice] == '+'):
                chiave = ramo.__str__() + "_pt_" + (indice+1).__str__()

                #Unisco tutte le stringhe nella lista in una sola, separandole da " + "
                valore = " + ".join(dict_ramo[chiavi_ramo[indice]])

                #Sostituisco alcune stringhe (mi servirà per l'eval)
                valore=valore.replace("P", "Point")
                valore=valore.replace("L", "Line")


                temp.add_child(Tree(name=chiave, gestures=valore.__str__()))
                indice+=1

        return radice

    def visit(self, tree):

        #Se il nodo ha figli, li ''visito'', altrimenti stampo che non ne ha
        if(tree.children!=[]):
            for figlio in tree.children:
                if(figlio.gestures!=None):
                    print(tree.name + " ha come figlio " + figlio.name + " e ha come gestures: " + figlio.gestures.__str__())
                else:
                    print(tree.name + " ha come figlio " + figlio.name + " e non ha gestures")
            for figlio in tree.children:
                figlio.visit(figlio)
        else:
            print(tree.name + " non ha figli")

    def createTreeDict(self, tree, gestures_hmms):


        #Creo l'hmm delle foglie dell'albero
        if (tree.children != []):
            for figlio in tree.children:
                figlio.createTreeDict(figlio, gestures_hmms)
        else:

            #ricostruisco la gesture_expression dal nome del nodo e dalle sue gestures (tramite eval)
            #gesture_exp={tree.name: [eval(tree.gestures)]}
            # gestures_hmms.append(ModelExpression.generatedModels(gesture_exp, num_states=6, spu=20))
            gestures_hmms.setdefault(tree.name, [eval(tree.gestures)])

    def recognizeByProbability(self, gesture, gesture_hmms, sample_frames):
        #gesture= gesture da riconoscere
        #gesture_hmms= il dizionario contenente gli hmms
        #sample_frame= numero di frames ogni quanto effettuare il riconoscimento

        results=[]

        unistroke_mode=True

        path0 = '/home/federico/PycharmProjects/deictic/repository/'+'deictic/1dollar-dataset/raw/'+gesture[0]+'/'

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

        #Confronto una coppia di punti del dataset con gli hmm dell'albero con la compare, che mi restituisce
        #il nome dell'hmm con la probabilità massima tra i vari hmms
        list_of_prob_fullGesture=[]
        someFrames=[]
        list_of_prob_someFrames=[]

        i=0
        y=0
        plot_frames=[]
        label=""
        prev_label=""
        onetime=True

        for ndarray, name in list_of_points:

            #Riconoscimento singolo file (ndarray)
            list_of_prob_fullGesture.append(Test.compare(sequence=ndarray, gesture_hmms=gesture_hmms,
                                                          return_log_probabilities=False))

            #Stampo il grafico finale
            if(someFrames!=[] and plot_frames!=[]):
                mp.title(label)
                mp.suptitle(name)
                label_changes-=1 #altrimenti cambia colore del punto

                #Stampo i punti residui
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

                mp.show()




            i=0 #Inizializzo il counter dei frame
            someFrames=[] #Inizializzo la lista contenente i frame quando cambio file

            label_changes = 0
            plot_frames=[]
            for ndarray1 in ndarray:
                i+=1


                #Riconoscimento attraverso un array di frames (viene incrementato di un frame per ogni ciclo)
                someFrames.append(ndarray1)
                plot_frames.append((ndarray1[0],ndarray1[1]))



                prev_label = label
                label=Test.compare(sequence=someFrames, gesture_hmms=gesture_hmms, return_log_probabilities=False)

                #Stampo grafico parziale
                a= (prev_label == label)
                if ((a==False) and label!=None):
                    for x,y in plot_frames:
                        if(label_changes==0):
                            mp.plot(x, y, 'bo')
                        if (label_changes == 1):
                            mp.plot(x, y, 'ro')
                        if (label_changes == 2):
                            mp.plot(x, y, 'go')
                        if (label_changes == 3):
                            mp.plot(x, y, 'yo')
                        if (label_changes == 4):
                            mp.plot(x, y, 'ro')
                    label_changes += 1

                    x,y=plot_frames[int(len(plot_frames)/2)]
                    mp.text(x,y, label)
                    plot_frames=[]

                #Stampa di aggiornamento (stampo in questo modo per sostituire sempre la riga attuale)
                stdout.write("\rCalcolo probabilità FILE: " + name + "  Frame " + i.__str__() +
                             "  Gesture riconosciuta: " + label.__str__())
                stdout.flush()

                #time.sleep(0.025)


                y+=1
                list_of_prob_someFrames.append(label)



        i=0

        #Result full è un dizionario gesture -> numero di riconoscimenti....full perchè appartiene al fullgesture
        result_full = {i: list_of_prob_fullGesture.count(i) for i in list_of_prob_fullGesture}
        print("\nLa gesture con il maggior numero di match nella modalità fullgesture è: ")
        max=-1

        #Cerco il massimo valore nel dizionario
        for label in list(result_full.keys()):
            if(max==-1):
                max=(label,(result_full[label]))
            if(result_full[label]>max[1]):
                max=(label,(result_full[label]))

        #Lo stampo
        print(max)
        print("Risultati complessivi: " + result_full.__str__())



    #Funzione extra (Riconoscimento di una gesture tramite confronto)
    def recognizeByCompare(self, tree, gesture):
        # Se il nodo ha figli, li ''visito'', altrimenti stampo che non ne ha
        if (tree.children != []):
            for figlio in tree.children:
                temp=(list(gesture.keys()))[0]
                temp=gesture[temp][0]
                if (figlio.gestures !=None):
                    gesture_figlio=[eval(figlio.gestures)][0]
                    if(temp.__str__() == gesture_figlio.__str__()):
                        print("Gesture riconosciuta come " + figlio.name)
            for figlio in tree.children:
                figlio.recognizeByCompare(figlio, gesture)


    def returnModels(self):
        pass




#### HINT ####
# hai presente il this di java? ecco, self è la stessa identica cosa. Per esempio, nella funzione recognizedByCompare
# non c'è bisogno di passare anche tree tra i parametri, perché tanto puoi accedere agli oggetti di quello stesso
# oggetto tramite la self: esempio, riga 328, anziché avere tree.children, avrai self.children. Non è sbagliato quello
# che fai, però è inutile. In pratica per ogni chiamata ti crei in memoria due volte lo stesso oggetto.

#### COSE DA FARE ####
# - 1) stavo controllando come creavi la lista di hmm. Ora cerchiamo di fare un passo in avanti, fai in modo che i nodi
#      foglia contengano anche i rispettivi hmm, oppure mettili nel nodo genitore. Poi valutiamo quale è la scelta migliore;
# - 2) la comparazione avviene attraverso un'altra funzione non contenuta in Tree.
# - 3) modifica la funzione "returnModels" in modo che restituisca tutte le hmm contenuti nell'albero (vedere poi
#      se dal nodo genitore o dai nodi figli);
# - 4) togli le parti che non servono e aggiusta i commenti.

#### MODIFICHE INSERITE ####
# - 1) ho creato il file __init__.py per poter richiamare Tree anche in altre parti della libreria.