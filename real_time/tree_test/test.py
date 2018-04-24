from gesture import *

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

    gestures_dictionary4={'Triangolo': [Point(0,0)+Line(0,1)+Line(1,1)+Line(0,0)],
                          'Rettangolo':[Point(0,0)+Line(0,1)+Line(1,1)+Line(0,1)+Line(0,0)],
                            'Delete': [Point(1,1)+Line(-1,-1)+Line(1,-1)+Line(-1,1)]}

    items = []
    tree=Tree("")
    list_of_operators=[]

    primitives_dictionary, list_of_operators = tree.gestureFactory(gestures_dictionary4)
    list_of_key=(list(primitives_dictionary.keys()))

    #print(primitives_dictionary)
    tree=tree.createTree(primitives_dictionary, 0, list_of_key, list_of_operators)

    tree.visit(tree)

main()