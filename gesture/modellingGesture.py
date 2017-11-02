from gesture import ClassifierFactory
from model import *
from topology import *

baseDir = '/home/sara/PycharmProjects/deictic/repository/'
#baseDir = '/home/ale/PycharmProjects/deictic/repository/'
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
        elif (exp == HmmFactory.TypeOperator.multistroke.name):
            expression = MDollarGestures.getModel(op1)
        elif (exp == HmmFactory.TypeOperator.unica.name):
            expression = UnicaGestures.getModel(op1)
        elif(exp == HmmFactory.TypeOperator.shrec.name):
            expression = Shrec.getModel(op1)
        # Adds gesture expression
        primitive = HmmFactory.factory(expression, self.n_states, self.n_samples)
        self.stack.append(primitive)


# Take a gesture type and return its complete model
class OneDollarGestures:

    class TypeGesture(Enum):
        triangle = 0
        x = 1
        rectangle = 2
        circle = 3
        check = 4
        caret = 5
        question_mark = 6
        left_sq_bracket = 7
        right_sq_bracket = 8
        v = 9
        delete_mark = 10
        left_curly_brace = 11
        right_curly_brace = 12
        star = 13
        pigtail = 14
        arrow = 15
        zig_zag = 16

    @staticmethod
    def getModel(type_gesture):
        definition = None

        # triangle
        if(type_gesture == OneDollarGestures.TypeGesture.triangle.name):
            definition = Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4)
        # x
        elif(type_gesture == OneDollarGestures.TypeGesture.x.name):
            definition = Point(0,0) + Line(3,-3) + Line(0,3) + Line(-3,-3)
        # rectangle
        elif(type_gesture == OneDollarGestures.TypeGesture.rectangle.name):
            definition = Point(0,0) + Line(0,-3) + Line(4,0) + Line(0, 3) + Line(-4,0)
        # circle
        elif(type_gesture == OneDollarGestures.TypeGesture.circle.name):
            definition = Point(0,0) + Arc(-3,-3, cw=False) + Arc(3,-3, cw=False) + Arc(3,3, cw=False) + Arc(-3,3, cw=False)
        # check
        elif(type_gesture == OneDollarGestures.TypeGesture.check.name):
            definition = Point(0,0) + Line(2, -2) + Line(4,6)
        # caret
        elif(type_gesture == OneDollarGestures.TypeGesture.caret.name):
            definition = Point(0,0) + Line(2,3) + Line(2,-3)
        # question mark
        elif(type_gesture == OneDollarGestures.TypeGesture.question_mark.name):
            definition = Point(0,0) + Arc(4,4) + Arc(4,-4) + Arc(-4,-4) + Arc(-2,-2, cw=False) + Arc(2, -2, cw=False)
        # left square bracket
        elif(type_gesture == OneDollarGestures.TypeGesture.left_sq_bracket.name):
            definition = Point(0,0) + Line(-4,0) + Line(0,-5) + Line(4,0)
        # right square bracket
        elif(type_gesture == OneDollarGestures.TypeGesture.right_sq_bracket.name):
            definition = Point(0,0) + Line(4,0) + Line(0, -5)  + Line(-4, 0)
        # v
        elif(type_gesture == OneDollarGestures.TypeGesture.v.name):
            definition = Point(0,0) + Line(2,-3) + Line(2,3)
        # delete_mark
        elif(type_gesture == OneDollarGestures.TypeGesture.delete_mark.name):
            definition = Point(0,0) + Line(2, -3) + Line(-2,0) + Line(2,3)
        # left curly brace
        elif(type_gesture == OneDollarGestures.TypeGesture.left_curly_brace.name):
            definition =  Point(0,0) + Arc(-5,-5, cw=False) + Arc(-3,-3)  + Arc(3,-3) +  Arc(5,-5,cw=False)
        # right curly brace
        elif(type_gesture == OneDollarGestures.TypeGesture.right_curly_brace.name):
            definition = Point(0,0) + Arc(5,-5) +  Arc(3,-3, cw=False) + Arc(-3,-3, cw=False) + Arc(-5,-5)
        # star
        elif(type_gesture == OneDollarGestures.TypeGesture.star.name):
            definition = Point(0,0) + Line(2,5) + Line(2, -5) + Line(-5, 3) + Line(6,0) + Line(-5, -3)
        # pigtail
        elif(type_gesture == OneDollarGestures.TypeGesture.pigtail.name):
            definition = Point(0,0) + Arc(3,3, cw=False) + Arc(-1,1, cw=False) + Arc(-1,-1, cw=False) + Arc(3, -3, cw=False)
        # arrow
        elif(type_gesture == OneDollarGestures.TypeGesture.arrow.name):
            definition = Point(0,0) + Line(6,4) + Line(-4,0) + Line(5,1) + Line(-1, -4)
        # zig_zag
        #if(type_gesture == OneDollarModels.TypeGesture.zig_zag.name):
            #definition = None

        return definition

class UnicaGestures(OneDollarGestures):

    @staticmethod
    def getModel(type_gesture):
        definition = super(UnicaGestures, UnicaGestures).getModel(type_gesture)
        if type_gesture == OneDollarGestures.TypeGesture.x.name:
            definition = Point(0, 0) + Line(3, 3) + Line(0, -3) + Line(-3, 3)
        elif type_gesture == OneDollarGestures.TypeGesture.delete_mark.name:
            definition = Point(0, 0) + Line(3, -3) + Line(-3, 0) + Line(3, 3)
        return definition



# Take a gesture type and return its complete model
class MDollarGestures:

    class TypeGesture(Enum):
        arrowhead = 0
        H = 1
        N = 2
        I = 3
        P = 4
        T = 5
        six_point_star = 6
        D = 7
        asterisk = 8
        exclamation_point = 9
        null = 10
        pitchfork = 11
        half_note = 12
        X = 13

    @staticmethod
    def getModel(type_gesture):
        definition = None
        # arrowhead
        if(type_gesture == MDollarGestures.TypeGesture.arrowhead.name):
            definition = Point(0, 0) + Line(6, 0) + Point(4, 2) + Line(2, -2) + Line(-2, -2)
        # h
        elif(type_gesture == MDollarGestures.TypeGesture.H.name):
            definition = Point(0, 4) + Line(0, -4) + Point(-1, 2) + Line(5, 0) + Point(4, 4) + Line(0, -4)
        # n
        elif(type_gesture == MDollarGestures.TypeGesture.N.name):
            definition = Point(0, 4) + Line(0, -4) + Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(0, -4)
        # i
        elif(type_gesture == MDollarGestures.TypeGesture.I.name):
            definition = Point(0, 4) + Line(4, 0) + Point(2, 4) + Line(0, -4) + Point(0, 0) + Line(4, 0)
        # p
        elif(type_gesture == MDollarGestures.TypeGesture.P.name):
            definition = Point(0, 0) + Line(0, -4) + Point(0, 0) + Arc(1, -1, cw=True) + Arc(-1, -1, cw=True)
        # t
        elif(type_gesture == MDollarGestures.TypeGesture.T.name):
            definition = Point(-2, 0) + Line(4, 0) + Point(0, 0) + Line(0, -4)
        # six point star
        elif(type_gesture == MDollarGestures.TypeGesture.six_point_star.name):
            definition = Point(0, 0.5) + Line(2, 2) + Line(2, -2) + Line(-4, 0) + Point(0, 2) + Line(4, 0) + Line(-2, -2) + Line(-2, 2)
        # d
        elif(type_gesture == MDollarGestures.TypeGesture.D.name):
            definition = Point(0, 0) + Line(0, 4) + Point(0, 4) + Arc(2, -2, cw=True) + Point(2, 2) + Arc(-2, -2, cw=True)
        # asterisk
        elif(type_gesture == MDollarGestures.TypeGesture.asterisk.name):
            definition = Point(4,3) + Line(-4,-3) + Point(0,3) + Line (4,-3) + Point(2,4) + Line(0, -4)
        # exclamation_point
        elif(type_gesture == MDollarGestures.TypeGesture.exclamation_point.name):
            definition = Point(0, 20) + Line(0, -19)+ Point(0, 1) + Line(0, -1)
        # null
        elif(type_gesture == MDollarGestures.TypeGesture.null.name):
            definition = Point(0,0) + Arc(-3,-3, cw=False) + Arc(3,-3, cw=False) + Arc(3,3, cw=False) + \
                         Arc(-3,3, cw=False) + Point(4,1) + Line(-8, -8)
        # pitchfork
        elif(type_gesture == MDollarGestures.TypeGesture.pitchfork.name):
            definition = Point(-2,4)+Arc(2,-2, cw=False) + Point(0,2)+Arc(2,2, cw=False) + Point(0,4)+Line(0,-4)
        # half note
        elif(type_gesture == MDollarGestures.TypeGesture.half_note.name):
            definition = Point(0,0)+Line(0,-4) + Point(0,-4)+Arc(-1,-1, cw=False) + Point(-1,-5)+Arc(1,1, cw=False)
        # x
        elif(type_gesture == MDollarGestures.TypeGesture.X.name):
            definition = Point(4,3) + Line(-4,-3) + Point(0,3) + Line (4,-3)

        return definition

class Shrec:

    class TypeGesture(Enum):

        gesture_2 = 2
        gesture_7 = 7
        gesture_8 = 8
        gesture_9 = 9
        gesture_10 = 10
        gesture_11 = 11
        gesture_12 = 12
        gesture_13 = 13
        gesture_14 = 14

    @staticmethod
    def getModel(type_gesture):
        definition = None

        # tap
        if (type_gesture == Shrec.TypeGesture.gesture_2.name):
            definition = Point(0, 0) +  Line(0, -4)
        # swipe right
        elif (type_gesture == Shrec.TypeGesture.gesture_7.name):
            definition = Point(-4, 2) + Line(4, 2)
        #swipe left
        elif (type_gesture == Shrec.TypeGesture.gesture_8.name):
            definition = Point(4, 2) + Line(-4, 2)
        #swipe up
        elif (type_gesture == Shrec.TypeGesture.gesture_9.name):
            definition = Point(4, -4) + Line(4, 4)
        #swipe down
        elif (type_gesture == Shrec.TypeGesture.gesture_10.name):
            definition = Point(4, 4) + Line(4, -4)
        #swipe x
        elif (type_gesture == Shrec.TypeGesture.gesture_11.name):
            definition = Point(-4, 4) + Line(-4, -4) + Line(4, -4) + Line(-4, 4)
        #swipe +
        elif (type_gesture == Shrec.TypeGesture.gesture_12.name):
            definition = Point(4, 4) + Line(4, -4) + Line(-4, 0) + Line(8, 0)
        #swipe v
        elif (type_gesture == Shrec.TypeGesture.gesture_13.name):
            definition = Point(-4, 4) + Line(-4, 0) + Line(0, 4)
        #shake
        elif (type_gesture == Shrec.TypeGesture.gesture_14.name):
            definition = Point(-4, 4) + Line(8, 4) + Line(-8, 4) + Line(8, 4) + Line(-8, 4)

        return definition