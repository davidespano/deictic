from gesture import *
from model.gestureModel import Point, Line, Arc
from test.test import Test


from real_time.tree_test.tree import Tree

def main():

    gestures_dictionary1 = {'triangle': [Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4)]}

    gestures_dictionary2 = {'triangle': [Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4)],
                           'rectangle': [Point(0,0) + Line(0,-3) + Line(4,0) + Line(0, 3) + Line(-4,0)]}

    gestures_dictionary3 = {'triangle': [Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4)],
                            'rectangle': [Point(0, 0) + Line(0, -3) + Line(4, 0) + Line(0, 3) + Line(-4, 0)],
                               'x': [Point(0,0) + Line(3,-3) + Line(0,3) + Line(-3,-3)]}

    items = []
    tree=Tree("")
    list_of_operators=[]

    primitives_dictionary, list_of_operators = tree.gestureFactory(gestures_dictionary3)
    list_of_key=(list(primitives_dictionary.keys()))

    #print(primitives_dictionary)
    tree=tree.createTree(primitives_dictionary, 0, list_of_key, list_of_operators)

    gestures_list={}
    gestures_hmms=[]

    tree.createTreeDict(tree, gestures_list)

    #Creo gli hmms dei nodi dell'albero attraverso quanto ottenuto con le precedenti funzioni
    gestures_hmms=ModelExpression.generatedModels(gestures_list)
    print("Albero:")
    tree.visit(tree)
    n_sample = 20
    list_gesture = [("triangle", 3 * n_sample),("rectangle", 4 * n_sample),("arrow", 4 * n_sample), ("caret", 2 * n_sample), ("circle", 4 * n_sample), ("check", 2 * n_sample),
                    ("delete_mark", 3 * n_sample),
                    ("left_curly_brace", 6 * n_sample), ("left_sq_bracket", 3 * n_sample), ("pigtail", 4 * n_sample),
                    ("question_mark", 4 * n_sample),
                     ("right_curly_brace", 6 * n_sample),
                    ("right_sq_bracket", 3 * n_sample), ("star", 5 * n_sample),
                     ("v", 2 * n_sample), ("x", 3 * n_sample)]
    print("")

    gesture=("triangle", 3 * n_sample)
    tree.recognizeByProbability(gesture=gesture, gesture_hmms=gestures_hmms, sample_frames=100)

    gesture={"triangle_pt2":[Point(0,0) + Line(-3,-4) + Line(6,0)]}
    gesture_trovata=False
    #tree.recognizeByCompare(tree=tree, gesture=gesture)
    print("end")


main()