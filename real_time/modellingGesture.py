from gesture import modellingGesture
from model import *
# Enum
from enum import Enum

class Parse(modellingGesture.Parse):
    def __gestureComponents(self, exp):
        """
            Manages parsing gesture components (from three different dataset)
        :param exp: the expression to parse
        :return: none
        """
        # Takes operand
        op1 = self.stack.pop()
        expression = None
        # Takes gesture expression
        if (exp == modellingGesture.HmmFactory.TypeOperator.unistroke.name):
            expression = OneDollarGestures.getModel(op1)
        # Adds gesture expressions
        gesture = modellingGesture.HmmFactory.factory(expression, self.n_states, self.n_samples)
        self.stack.append(gesture)
        #for exp in expression:
        #    primitive = modellingGesture.HmmFactory.factory(exp, self.n_states, self.n_samples)
        #    self.stack.append(primitive)



class OneDollarGestures(modellingGesture.OneDollarGestures):

    class TypeGesture(Enum):
        # Primitives
        v_1 = 1
        v_2 = 2
        x_1 = 3
        x_2 = 4
        x_3 = 5
        rectangle_1 = 6
        rectangle_2 = 7
        rectangle_3 = 8
        rectangle_4 = 9
        star_1 = 10
        star_2 = 11
        star_3 = 12
        star_4 = 13
        star_5 = 14

    @staticmethod
    def getModel(type_gesture):
        # Primitives V
        if(type_gesture == OneDollarGestures.TypeGesture.v_1.name):
            #definition = [Point(0,0) + Line(2,-3), Line(2,3), Point(0,0) + Line(2,-3) + Line(2,3)]
            definition = Point(0,0)+Line(2,-3)
        elif(type_gesture == OneDollarGestures.TypeGesture.v_2.name):
            definition = Point(0,0) + Line(2,-3) + Line(2,3)
        # X
        elif(type_gesture == OneDollarGestures.TypeGesture.x_1.name):
            definition = Point(0, 0) + Line(3, -3)
        elif (type_gesture == OneDollarGestures.TypeGesture.x_2.name):
            definition = Point(0, 0) + Line(3, -3) + Line(0, 3)
        elif (type_gesture == OneDollarGestures.TypeGesture.x_3.name):
            definition = Point(0, 0) + Line(3, -3) + Line(0, 3) + Line(-3,-3)
        # Rectangle
        elif(type_gesture == OneDollarGestures.TypeGesture.rectangle_1.name):
            definition = Point(0, 0) + Line(0, -3)
        elif (type_gesture == OneDollarGestures.TypeGesture.rectangle_2.name):
            definition = Point(0, 0) + Line(0, -3) + Line(4, 0)
        elif (type_gesture == OneDollarGestures.TypeGesture.rectangle_3.name):
            definition = Point(0, 0) + Line(0, -3) + Line(4, 0) + Line(0, 3)
        elif (type_gesture == OneDollarGestures.TypeGesture.rectangle_4.name):
            definition = Point(0, 0) + Line(0, -3) + Line(4, 0) + Line(0, 3) + Line(-4,0)
        # Star
        elif (type_gesture == OneDollarGestures.TypeGesture.star_1.name):
            definition = Point(0, 0) + Line(2, 5)
        elif (type_gesture == OneDollarGestures.TypeGesture.star_2.name):
            definition = Point(0, 0) + Line(2, 5) + Line(2, -5)
        elif (type_gesture == OneDollarGestures.TypeGesture.star_3.name):
            definition = Point(0, 0) + Line(2, 5) + Line(2, -5) + Line(-5, 3)
        elif (type_gesture == OneDollarGestures.TypeGesture.star_4.name):
            definition = Point(0, 0) + Line(2, 5) + Line(2, -5) + Line(-5, 3) + Line(6, 0)
        elif (type_gesture == OneDollarGestures.TypeGesture.star_5.name):
            definition = Point(0, 0) + Line(2, 5) + Line(2, -5) + Line(-5, 3) + Line(6, 0) + Line(-5, -3)

        else:
            definition = super(OneDollarGestures, OneDollarGestures).getModel(type_gesture)

        return  definition