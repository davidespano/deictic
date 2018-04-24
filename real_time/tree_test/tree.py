import re
import more_itertools as mit


class Tree(object):

    def __init__(self, name, children=[]):
        if(isinstance(name, str) or isinstance(name, tuple)):
            self.name = name.__str__()
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
                chiave = chiavi_ramo[indice].__str__()
                valore = dict_ramo[chiavi_ramo[indice]].__str__()
                coppia = (chiave, valore)

                temp.add_child(Tree(coppia))
                indice+=1

        return radice


    def visit(self, tree):

        #Se il nodo ha figli, li ''visito'', altrimenti stampo che non ne ha
        if(tree.children!=[]):
            for figlio in tree.children:
                print(tree.name + " ha come figlio " + figlio.name)
            for figlio in tree.children:
                figlio.visit(figlio)
        else:
            print(tree.name + " non ha figli")




