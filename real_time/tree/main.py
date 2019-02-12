from gesture import *
from model.gestureModel import Point, Line, Arc
from real_time.tree import recognition as re
from real_time.tree.tree import Tree

def main():

    ####################################################################################################################
    ####                                               TUTORIAL                                                     ####
    ####################################################################################################################
    #                                                                                                                  #
    #Il metodo  ##gestureFactory## splitta la gesture nei vari nodi dell'albero (per adesso gestisce solo la sequence) #
    #                                                                                                                  #
    #Prende in input un DIZIONARIO che ha come chiavi il nome della gesture, e come valore una compositeExpression     #
    #                                                                                                                  #
    #La funzione restituisce un dizionario contenente la scomposizione della gesture (nome gesture: scomposizione)     #
    #nella diverse primitive e una lista degli operatori presenti nella compositeExpression                            #
    #(Quest'ultima serviva per la scomposizione di espressione contenenti anche altri operatori al di fuori della      #
    #sequence (+), ma non ho mai finito di implementarla                                                               #
    #                                                                                                                  #
    #                                      #######################                                                     #
    #                                                                                                                  #
    #Il metodo ##createTree## crea un oggetto di tipo Tree e lo popola con i diversi nodi corrispondenti alle gesture e#
    #alle loro scomposizioni. Ogni nodo può contenere una lista dei nodi figli, il proprio hmm, l'hmm dei figli e la   #
    #propria gesture. Va richiamata a partire da un oggetto di tipo Tree inizializzato come segue: tree=Tree("").      #
    #Ciò permette di creare la radice dell'albero e di richiamare tutti i metodi per modellarlo                        #
    #                                                                                                                  #
    #Prende in input un dizionario contenente la scomposizione della gesture nelle diverse primitive, una lista delle  #
    #chiavi di questo dizionario e la lista degli operatori (stesso discorso della precedente funzione).               #
    #                                                                                                                  #
    #Ovviamente, restituisce un oggetto di tipo Tree                                                                   #
    #                                                                                                                  #
    ##                                     #######################                                                     #
    #                                                                                                                  #
    #Il metodo ##createTreeDict## crea il dizionario dell'albero passato in input, inserendo come chiave il nome della #
    #sottogesture e come valore la sua compositeExp. Restituisce il dizionario creato                                  #
    #                                                                                                                  #
    #                                      #######################                                                     #
    #                                                                                                                  #
    #Il metodo ##visit## effettua la visita in ampiezza dell'albero inserito in input                                  #
    #                                                                                                                  #
    #                                      #######################                                                     #
    #                                                                                                                  #
    #Il metodo ##returnModels## restituisce un dizionario corrispondente all'hmm di tutti i nodi dell'albero           #
    #                                                                                                                  #
    #                                      #######################                                                     #
    #                                                                                                                  #
    #Il metodo ##recognizeByProbability## stampa a video il grafico della gesture riconosciuta framebyframe e indica,  #
    #inoltre, quale è stata riconosciuta mediante il riconoscimento filebyfile                                         #
    #                                                                                                                  #
    #Prende in input una gesture (es: ("triangle", 3 * n_sample) ), l'hmm dell'albero, e un bool che permette di       #
    #scegliere se visualizzare il grafico o meno (in ogni caso, stampa su console il riconoscimento in tempo reale     #
    #                                                                                                                  #
    ###################################################################################################################


    gestures_dictionary1 = {'triangle': [Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4)]}

    gestures_dictionary2 = {'triangle': [Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4)],
                           'rectangle': [Point(0,0) + Line(0,-3) + Line(4,0) + Line(0, 3) + Line(-4,0)]}

    gestures_dictionary3 = {
                                'arrow': [
                                    Point(0, 0) + Line(6, 4) + Line(-4, 0) + Line(5, 1) + Line(-1, -4)
                                ],
                                'caret': [
                                    Point(0, 0) + Line(2, 3) + Line(2, -3)
                                ],
                                'check': [
                                    Point(0, 0) + Line(2, -2) + Line(4, 6)
                                ],
                                'circle': [
                                    Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False)
                                ],
                                'delete_mark': [
                                    Point(0, 0) + Line(2, -3) + Line(-2, 0) + Line(2, 3)
                                ],
                                'left_curly_brace': [
                                    Point(0, 0) + Arc(-5, -5, cw=False) + Arc(-3, -3) + Arc(3, -3) + Arc(5, -5, cw=False)
                                ],
                                'left_sq_bracket': [
                                    Point(0, 0) + Line(-4, 0) + Line(0, -5) + Line(4, 0)
                                ],
                                'pigtail':[
                                    Point(0,0) + Arc(3,3, cw=False) + Arc(-1,1, cw=False) + Arc(-1,-1, cw=False) + Arc(3, -3, cw=False)
                                ],
                                'question_mark':[
                                    Point(0,0) + Arc(4,4) + Arc(4,-4) + Arc(-4,-4) + Arc(-2,-2, cw=False) + Arc(2, -2, cw=False)
                                ],
                                'rectangle':[
                                    Point(0,0) + Line(0,-3) + Line(4,0) + Line(0, 3) + Line(-4,0)
                                ],
                                'right_curly_brace':[
                                    Point(0,0) + Arc(5,-5) +  Arc(3,-3, cw=False) + Arc(-3,-3, cw=False) + Arc(-5,-5)
                                ],
                                'right_sq_bracket':[
                                    Point(0,0) + Line(4,0) + Line(0, -5) + Line(-4, 0)
                                ],
                                'star':[
                                    Point(0,0) + Line(2,5) + Line(2, -5) + Line(-5, 3) + Line(6,0) + Line(-5, -3)
                                ],
                                'triangle':[
                                    Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4)
                                ],
                                'v':[
                                    Point(0,0) + Line(2,-3) + Line(2,3)
                                ],
                                'x':[
                                    Point(0,0) + Line(3,-3) + Line(0,3) + Line(-3,-3)
                                ],
                            }

    items = []

    tree=Tree(gesture_exp=gestures_dictionary3) #Creazione di un oggetto Tree, inserendo il dizionario sopra definito

    gestures_list=tree.createTreeDict()

    print("\nAlbero:")


    tree.visit()

    n_sample = 20
    list_gesture = [("triangle", 3 * n_sample),("rectangle", 4 * n_sample),("arrow", 4 * n_sample), ("caret", 2 * n_sample), ("circle", 4 * n_sample), ("check", 2 * n_sample),
                    ("delete_mark", 3 * n_sample),
                    ("left_curly_brace", 6 * n_sample), ("left_sq_bracket", 3 * n_sample), ("pigtail", 4 * n_sample),
                    ("question_mark", 4 * n_sample),
                     ("right_curly_brace", 6 * n_sample),
                    ("right_sq_bracket", 3 * n_sample), ("star", 5 * n_sample),
                     ("v", 2 * n_sample), ("x", 3 * n_sample)]
    print("")

    #gesture_hmms contiene tutti gli hmm dell'albero
    gestures_hmms=tree.returnModels()


    ##########################
    ##       LEGGIMI        ##
    ##########################
    #           |            #
    #           |            #
    #           V            #
    #Se vuoi eseguire il riconoscimento framebyframe, decommenta recognizeByPRobability
    #Se vuoi plottare i frames dei file originali delle gesture, decommenta plotCsvFile
    #Solo una delle due deve essere decommentata, altrimenti mischi le stampe


    for gesture in list_gesture:
        #re.recognizeByProbability(gesture=gesture, gesture_hmms=gestures_hmms, enable_show=True)
        pass
        #re.plotCsvFile(gesture)


    print("\n################\n")


    #Ottengo il dizionario nomefile: frame della scomposizione
    changePrimitivesDict = re.readChangePrimitivesFile()


    print("end")


main()