from gesture import *
from model.gestureModel import Point, Line, Arc


from real_time.tree_test.tree import Tree

def main():

    gestures_dictionary0 = {'test0':
                                [Point(0, 0) + Line(-1, 1) + Line(-1, 2)]
                            }

    '''gestures_dictionary1 = {'test1':
                               [Point(0, 0) + Line(-1, 1) + Line(-1, 2) | Point(0, 1) + Line(1, -1) + Line(1, -2) ]
                           }

    gestures_dictionary2 = {'test2':
                                [Point(0, 0) + Line(-1, 1) + Line(-1, 2) * Point(0, 1) + Line(1, -1) + Line(1, -2) ]
                            }

    gestures_dictionary3= {'test3':
                                [(Line(1,2) + Line(3,4)) * (Line(5,6) + Line(7,8)) | (Line(-1,-2) + Line(-3,-4)) * (Line(-5,-6) + Line(-7,-8))]
                            }'''

    gestures_dictionary4={'triangle': [Point(0,0)+Line(1,2)+Line(3,4)+Line(0,1)],
                          'rectangle':[Point(0,0)+Line(0,1)+Line(1,1)+Line(0,1)+Line(0,1)],
                            'delete': [Point(1,1)+Line(-1,-1)+Line(1,-1)+Line(-1,1)]}

    gestures_dictionary5 = {'triangle': [Point(0,0)+Line(1,2)+Line(3,4)+Line(0,1)]}


    items = []
    tree=Tree("")
    list_of_operators=[]

    primitives_dictionary, list_of_operators = tree.gestureFactory(gestures_dictionary5)
    list_of_key=(list(primitives_dictionary.keys()))

    #print(primitives_dictionary)
    tree=tree.createTree(primitives_dictionary, 0, list_of_key, list_of_operators)

    tree.visit(tree)

    gestures_list={}
    gestures_hmms_auto=[]
    gestures_hmms_manual=[]



    tree.createTreeDict(tree, gestures_list)

    #Creo gli hmms dei nodi dell'albero attraverso quanto ottenuto con le precedenti funzioni
    gestures_hmms_auto=ModelExpression.generatedModels(gestures_list)

    #Ricalcolo gli hmms a mano, per confrontare il risultato
    gesture1 = {'triangle_pt_1': [Point(0, 0) + Line(1, 2)]}
    gesture2 = {'triangle_pt_2': [Point(0, 0) + Line(1, 2) + Line(3, 4)]}
    gesture3 = {'triangle_pt_3': [Point(0, 0) + Line(1, 2) + Line(3, 4) + Line(0, 1)]}

    hmm1 = ModelExpression.generatedModels(gesture1)
    hmm2 = ModelExpression.generatedModels(gesture2)
    hmm3 = ModelExpression.generatedModels(gesture3)

    #probabilities_list=tree.calculateProbability(20, gestures_hmms)

    print("end")


main()