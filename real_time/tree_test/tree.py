import re
import more_itertools as mit
from gesture import *
from test.test import Test
import matplotlib.pyplot as mp
from sys import stdout
import time

class Tree(object):

    def __init__(self, name, gestures=None, children=[], hmm=None, children_hmm=None):
        if(isinstance(name, str)):
            self.name = name.__str__()
            self.gestures=gestures
            self.children = []
            self.hmm= hmm
            self.children_hmm=children_hmm


    def add_child(self, node):
        if (isinstance(node, Tree)):
            self.children.append(node)

    # Splitta la gesture nei vari nodi dell'albero (per adesso gestisce solo le sequenze)
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

    def createTree(self, dictionary, chiavi, operators, indice=0):

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
                valore=valore.replace("A", "Arc")
                gesture_list={chiave:[eval(valore)]}

                stdout.write("\rCalcolo hmm nodo:" + chiave.__str__() )
                stdout.flush()

                temp.add_child(Tree(name=chiave, gestures=valore.__str__(), children=[], hmm=ModelExpression.generatedModels(gesture_list)))
                indice+=1

            if (temp.children == []):
                temp.children_hmm = None
            else:
                children_hmm = {}
                for figlio in temp.children:
                    children_hmm.update(figlio.hmm)
                temp.children_hmm = children_hmm

        return radice

    def visit(self, tree):

        #Se il nodo ha figli, li ''visito'', altrimenti stampo che non ne ha
        if(tree.children!=[]):
            for figlio in tree.children:
                if(figlio.gestures!=None):
                    print(tree.name + " ha come figlio " + figlio.name + " e come gestures: " + figlio.gestures.__str__())
                else:
                    print(tree.name + " ha come figlio " + figlio.name + " e non ha gestures")
            for figlio in tree.children:
                figlio.visit(figlio)
        else:
            print(tree.name + " non ha figli")

    def createTreeDict(self, tree, gestures_hmms={}):


        #Creo l'hmm delle foglie dell'albero
        if (tree.children != []):
            for figlio in tree.children:
                figlio.createTreeDict(figlio, gestures_hmms)
            return gestures_hmms
        else:

            #ricostruisco la gesture_expression dal nome del nodo e dalle sue gestures (tramite eval)
            #gesture_exp={tree.name: [eval(tree.gestures)]}
            # gestures_hmms.append(ModelExpression.generatedModels(gesture_exp, num_states=6, spu=20))
            gestures_hmms.setdefault(tree.name, [eval(tree.gestures)])

    def returnModels(self, tree, hmm={}):
        if (tree.children != []):
            for figlio in tree.children:
                if (figlio.hmm != None):
                    hmm.update(figlio.hmm)

                figlio.returnModels(figlio, hmm)
            return hmm




#### HINT ####
# hai presente il this di java? ecco, self è la stessa identica cosa. Per esempio, nella funzione recognizedByCompare
# non c'è bisogno di passare anche tree tra i parametri, perché tanto puoi accedere agli oggetti di quello stesso
# oggetto tramite la self: esempio, riga 328, anziché avere tree.children, avrai self.children. Non è sbagliato quello
# che fai, però è inutile. In pratica per ogni chiamata ti crei in memoria due volte lo stesso oggetto.
