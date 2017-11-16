from gesture import datasetExpressions
from model import *
# Enum
from enum import Enum

from gesture import ClassifierFactory
from model import *
from topology import *

#baseDir = '/home/sara/PycharmProjects/deictic/repository/'
baseDir = '/home/ale/PycharmProjects/deictic/repository/'
#baseDir = '/Users/davide/PycharmProjects/deictic/repository/'
trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'

class HmmFactory:

    class TypeOperator(Enum):
        disabling = 0
        iterative = 1
        sequence = 2
        parallel = 3
        choice = 4
        unistroke = 5
        multistroke = 6
        unica = 7
        shrec = 8

    @staticmethod
    def factory(gesture, n_states, n_samples):
        factory = ClassifierFactory()
        factory.setLineSamplesPath(trainingDir)
        factory.setClockwiseArcSamplesPath(arcClockWiseDir)
        factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
        factory.states = n_states
        factory.spu = n_samples
        model, edges = factory.createClassifier(gesture)
        return model

class Parse:

    def __init__(self, n_states=6, n_samples=20):
        # Num of states and num of samples #
        if isinstance(n_states, int):
            self.n_states = n_states
        if isinstance(n_samples, int):
            self.n_samples = n_samples
        # Stack for parsing #
        self.stack = []

    def parseExpression(self, expression):
        """
            Manages parsing expression.
        :param expression: the expression to parse
        :return: the obtained hmm
        """
        # Split expression
        splitted = expression.split('-');
        rev = reversed(splitted)

        for exp in rev:
            # Gestit binary operators
            if exp in [HmmFactory.TypeOperator.disabling.name, HmmFactory.TypeOperator.sequence.name,
                       HmmFactory.TypeOperator.parallel.name, HmmFactory.TypeOperator.choice.name]:
                self.__binaryOperators(exp)
            # Gestit unary operators
            if exp in [HmmFactory.TypeOperator.iterative.name]:
                self.__unaryOperators(exp)
            # Gesture Expression
            if exp in [HmmFactory.TypeOperator.unistroke.name,
                       HmmFactory.TypeOperator.multistroke.name,
                       HmmFactory.TypeOperator.unica.name,
                       HmmFactory.TypeOperator.shrec.name]:
                self.__gestureComponents(exp)
            else:
                # Add exp expression
                self.stack.append(exp)
        return self.stack.pop()

    def __binaryOperators(self):
        """
            Manages parsing gestit binary/multiple operators (disabling, sequence, parallel and choice).
        :param exp: the expression to parse
        :return: none
        """
        # Take operands
        new_hmm = self.stack.pop()
        while self.stack:
            op = self.stack.pop()
            # Sequence
            if op == HmmFactory.TypeOperator.sequence.name:
                new_hmm, seq = HiddenMarkovModelTopology.sequence([new_hmm, op], [])
            # Parallel
            elif op == HmmFactory.TypeOperator.parallel.name:
                new_hmm, seq = HiddenMarkovModelTopology.parallel(new_hmm, op, [])
            # Choice
            elif op == HmmFactory.TypeOperator.choice.name:
                new_hmm, seq = HiddenMarkovModelTopology.choice([new_hmm, op], [])
        # Add new hmm
        self.stack.append(new_hmm)

    def __unaryOperators(self, exp):
        """
            Manages parsing gestit unary operators (iterative)
        :param exp: the expression to parse
        :return: none
        """
        # Take operand
        op1 = self.stack.pop()
        # Iterative
        if (exp == HmmFactory.TypeOperator.iterative.name):
            new_hmm, seq = HiddenMarkovModelTopology.iterative(op1, [])
        # Add hmm
        self.stack.append(new_hmm)

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
        if (exp == HmmFactory.TypeOperator.unistroke.name):
            expression = OneDollarGestures.getModel(op1)
        # Adds gesture expressions
        gesture = HmmFactory.factory(expression, self.n_states, self.n_samples)
        self.stack.append(gesture)
        # for exp in expression:
        #    primitive = modellingGesture.HmmFactory.factory(exp, self.n_states, self.n_samples)
        #    self.stack.append(primitive)


class OneDollarGestures():
    class TypeGesture(Enum):
        # Primitives
        circle_1 = 0,
        circle_2 = 1,
        circle_3 = 2,
        circle_4 = 3,
        rectangle_1 = 4,
        rectangle_2 = 5,
        rectangle_3 = 6,
        rectangle_4 = 7,
        star_1 = 8,
        star_2 = 9,
        star_3 = 10,
        star_4 = 11,
        star_5 = 12,
        triangle_1 = 13,
        triangle_2 = 14,
        triangle_3 = 15,
        v_1 = 16,
        v_2 = 17,
        x_1 = 18,
        x_2 = 19,
        x_3 = 20,
        check_1 = 21,
        check_2 = 22,
        caret_1 = 23,
        caret_2 = 24,
        arrow_1 = 25,
        arrow_2 = 26,
        arrow_3 = 27,
        arrow_4 = 28,
        left_sq_bracket_1 = 29,
        left_sq_bracket_2 = 30,
        left_sq_bracket_3 = 31,
        right_sq_bracket_1 = 32,
        right_sq_bracket_2 = 33,
        right_sq_bracket_3 = 34,
        delete_mark_1 = 35,
        delete_mark_2 = 36,
        delete_mark_3 = 37,
        left_curly_brace_1 = 38,
        left_curly_brace_2 = 39,
        left_curly_brace_3 = 40,
        left_curly_brace_4 = 41,
        right_curly_brace_1 = 42,
        right_curly_brace_2 = 43,
        right_curly_brace_3 = 44,
        right_curly_brace_4 = 45,
        pigtail_1 = 46,
        pigtail_2 = 47,
        pigtail_3 = 48,
        pigtail_4 = 49,
        question_mark_1 = 50,
        question_mark_2 = 51,
        question_mark_3 = 52,
        question_mark_4 = 53,
        question_mark_5 = 54,

    @staticmethod
    def getModel(type_gesture):
        # Primitives
        # Arrow
        if (type_gesture == OneDollarGestures.TypeGesture.arrow_1.name):
            definition = Point(0, 0) + Line(6, 4)
        elif (type_gesture == OneDollarGestures.TypeGesture.arrow_2.name):
            definition = Point(0, 0) + Line(6, 4) + Line(-4, 0)
        elif (type_gesture == OneDollarGestures.TypeGesture.arrow_3.name):
            definition = Point(0, 0) + Line(6, 4) + Line(-4, 0) + Line(5, 1)
        elif (type_gesture == OneDollarGestures.TypeGesture.arrow_4.name):
            definition = Point(0, 0) + Line(6, 4) + Line(-4, 0) + Line(5, 1) + Line(-1, -4)

        # Circle
        elif (type_gesture == OneDollarGestures.TypeGesture.circle_1.name):
            definition = Point(0, 0) + Arc(-3, -3, cw=False)
        elif (type_gesture == OneDollarGestures.TypeGesture.circle_2.name):
            definition = Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False)
        elif (type_gesture == OneDollarGestures.TypeGesture.circle_3.name):
            definition = Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False)
        elif (type_gesture == OneDollarGestures.TypeGesture.circle_4.name):
            definition = Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Arc(-3, 3,
                                                                                                                cw=False)

        # Check
        elif (type_gesture == OneDollarGestures.TypeGesture.check_1.name):
            definition = Point(0, 0) + Line(2, -2)
        elif (type_gesture == OneDollarGestures.TypeGesture.check_2.name):
            definition = Point(0, 0) + Line(2, -2) + Line(4, 6)

        # Caret
        elif (type_gesture == OneDollarGestures.TypeGesture.caret_1.name):
            definition = Point(0, 0) + Line(2, 3)
        elif (type_gesture == OneDollarGestures.TypeGesture.caret_2.name):
            definition = Point(0, 0) + Line(2, 3) + Line(2, -3)

        # Delete Mark
        elif (type_gesture == OneDollarGestures.TypeGesture.delete_mark_1.name):
            definition = Point(0, 0) + Line(2, -3)
        elif (type_gesture == OneDollarGestures.TypeGesture.delete_mark_2.name):
            definition = Point(0, 0) + Line(2, -3) + Line(-2, 0)
        elif (type_gesture == OneDollarGestures.TypeGesture.delete_mark_3.name):
            definition = Point(0, 0) + Line(2, -3) + Line(-2, 0) + Line(2, 3)

        # left curly brace
        elif (type_gesture == OneDollarGestures.TypeGesture.left_curly_brace_1.name):
            definition = Point(0, 0) + Arc(-5, -5, cw=False)
        elif (type_gesture == OneDollarGestures.TypeGesture.left_curly_brace_2.name):
            definition = Point(0, 0) + Arc(-5, -5, cw=False) + Arc(-3, -3)
        elif (type_gesture == OneDollarGestures.TypeGesture.left_curly_brace_3.name):
            definition = Point(0, 0) + Arc(-5, -5, cw=False) + Arc(-3, -3) + Arc(3, -3)
        elif (type_gesture == OneDollarGestures.TypeGesture.left_curly_brace_4.name):
            definition = Point(0, 0) + Arc(-5, -5, cw=False) + Arc(-3, -3) + Arc(3, -3) + Arc(5, -5, cw=False)

        # Left square bracket
        elif (type_gesture == OneDollarGestures.TypeGesture.left_sq_bracket_1.name):
            definition = Point(0, 0) + Line(-4, 0)
        elif (type_gesture == OneDollarGestures.TypeGesture.left_sq_bracket_2.name):
            definition = Point(0, 0) + Line(-4, 0) + Line(0, -5)
        elif (type_gesture == OneDollarGestures.TypeGesture.left_sq_bracket_3.name):
            definition = Point(0, 0) + Line(-4, 0) + Line(0, -5) + Line(4, 0)

        # Pigtail
        elif (type_gesture == OneDollarGestures.TypeGesture.pigtail_1.name):
            definition = Point(0, 0) + Arc(3, 3, cw=False)
        elif (type_gesture == OneDollarGestures.TypeGesture.pigtail_2.name):
            definition = Point(0, 0) + Arc(3, 3, cw=False) + Arc(-1, 1, cw=False)
        elif (type_gesture == OneDollarGestures.TypeGesture.pigtail_3.name):
            definition = Point(0, 0) + Arc(3, 3, cw=False) + Arc(-1, 1, cw=False) + Arc(-1, -1, cw=False)
        elif (type_gesture == OneDollarGestures.TypeGesture.pigtail_4.name):
            definition = Point(0, 0) + Arc(3, 3, cw=False) + Arc(-1, 1, cw=False) + Arc(-1, -1, cw=False) + Arc(3, -3,
                                                                                                                cw=False)

        # question mark
        elif (type_gesture == OneDollarGestures.TypeGesture.question_mark_1.name):
            definition = Point(0, 0) + Arc(4, 4)
        elif (type_gesture == OneDollarGestures.TypeGesture.question_mark_2.name):
            definition = Point(0, 0) + Arc(4, 4) + Arc(4, -4)
        elif (type_gesture == OneDollarGestures.TypeGesture.question_mark_3.name):
            definition = Point(0, 0) + Arc(4, 4) + Arc(4, -4) + Arc(-4, -4)
        elif (type_gesture == OneDollarGestures.TypeGesture.question_mark_4.name):
            definition = Point(0, 0) + Arc(4, 4) + Arc(4, -4) + Arc(-4, -4) + Arc(-2, -2, cw=False)
        elif (type_gesture == OneDollarGestures.TypeGesture.question_mark_5.name):
            definition = Point(0, 0) + Arc(4, 4) + Arc(4, -4) + Arc(-4, -4) + Arc(-2, -2, cw=False) + Arc(2, -2,
                                                                                                          cw=False)

        # Rectangle
        elif (type_gesture == OneDollarGestures.TypeGesture.rectangle_1.name):
            definition = Point(0, 0) + Line(0, -3)
        elif (type_gesture == OneDollarGestures.TypeGesture.rectangle_2.name):
            definition = Point(0, 0) + Line(0, -3) + Line(4, 0)
        elif (type_gesture == OneDollarGestures.TypeGesture.rectangle_3.name):
            definition = Point(0, 0) + Line(0, -3) + Line(4, 0) + Line(0, 3)
        elif (type_gesture == OneDollarGestures.TypeGesture.rectangle_4.name):
            definition = Point(0, 0) + Line(0, -3) + Line(4, 0) + Line(0, 3) + Line(-4, 0)

        # Right curly brace
        elif (type_gesture == OneDollarGestures.TypeGesture.right_curly_brace_1.name):
            definition = Point(0, 0) + Arc(5, -5)
        elif (type_gesture == OneDollarGestures.TypeGesture.right_curly_brace_2.name):
            definition = Point(0, 0) + Arc(5, -5) + Arc(3, -3, cw=False)
        elif (type_gesture == OneDollarGestures.TypeGesture.right_curly_brace_3.name):
            definition = Point(0, 0) + Arc(5, -5) + Arc(3, -3, cw=False) + Arc(-3, -3, cw=False)
        elif (type_gesture == OneDollarGestures.TypeGesture.right_curly_brace_4.name):
            definition = Point(0, 0) + Arc(5, -5) + Arc(3, -3, cw=False) + Arc(-3, -3, cw=False) + Arc(-5, -5)

        # Right square bracket
        elif (type_gesture == OneDollarGestures.TypeGesture.right_sq_bracket_1.name):
            definition = Point(0, 0) + Line(4, 0)
        elif (type_gesture == OneDollarGestures.TypeGesture.right_sq_bracket_2.name):
            definition = Point(0, 0) + Line(4, 0) + Line(0, -5)
        elif (type_gesture == OneDollarGestures.TypeGesture.right_sq_bracket_3.name):
            definition = Point(0, 0) + Line(4, 0) + Line(0, -5) + Line(-4, 0)

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

        # Triangle
        elif (type_gesture == OneDollarGestures.TypeGesture.triangle_1.name):
            definition = Point(0, 0) + Line(-3, -4)
        elif (type_gesture == OneDollarGestures.TypeGesture.triangle_2.name):
            definition = Point(0, 0) + Line(-3, -4) + Line(6, 0)
        elif (type_gesture == OneDollarGestures.TypeGesture.triangle_3.name):
            definition = Point(0, 0) + Line(-3, -4) + Line(6, 0) + Line(-3, 4)

        # V
        elif (type_gesture == OneDollarGestures.TypeGesture.v_1.name):
            definition = Point(0, 0) + Line(2, -3)
        elif (type_gesture == OneDollarGestures.TypeGesture.v_2.name):
            definition = Point(2, -3) + Line(2, 3)

        # X
        elif (type_gesture == OneDollarGestures.TypeGesture.x_1.name):
            definition = Point(0, 0) + Line(3, -3)
        elif (type_gesture == OneDollarGestures.TypeGesture.x_2.name):
            definition = Point(0, 0) + Line(3, -3) + Line(0, 3)
        elif (type_gesture == OneDollarGestures.TypeGesture.x_3.name):
            definition = Point(0, 0) + Line(3, -3) + Line(0, 3) + Line(-3, -3)

        else:
            definition = super(OneDollarGestures, OneDollarGestures).getModel(type_gesture)

        return definition