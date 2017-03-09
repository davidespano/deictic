from gesture import *
from model import *

baseDir = '/home/alessandro/PycharmProjects/deictic/repository/'
#baseDir = '/Users/davide/Google Drive/Dottorato/Software/python/hmmtest/repository/'
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

    @staticmethod
    def factory(gesture, n_states, n_samples):
        factory = ClassifierFactory()
        factory.setLineSamplesPath(trainingDir)
        factory.setClockwiseArcSamplesPath(arcClockWiseDir)
        factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
        factory.states = n_states
        factory.spu = n_samples
        model, edges = factory.createClassifier(gesture)
        return  model

class Parse:
    def __init__(self, n_states=6, n_samples=20):
        super()
        if isinstance(n_states, int):
            self.n_states = n_states
        if isinstance(n_samples, int):
            self.n_samples = n_samples

    def parse_expression(self, expression):
        # Split expression
        print(expression)
        expression = reversed(expression.split('-'))

        stack = []
        for exp in expression:
            # Binary operators
            if exp in [HmmFactory.TypeOperator.disabling.name, HmmFactory.TypeOperator.sequence.name,
                       HmmFactory.TypeOperator.parallel.name, HmmFactory.TypeOperator.choice.name]:
                # Take operands
                op1 = stack.pop()
                op2 = stack.pop()
                # Disabling
                if exp == HmmFactory.TypeOperator.disabling.name:
                    hmm = HmmFactory.factory(op1 % op2, self.n_states, self.n_samples)
                # Sequence
                elif exp == HmmFactory.TypeOperator.sequence.name:
                    hmm = HmmFactory.factory(op1 + op2, self.n_states, self.n_samples)
                # Parallel
                elif exp == HmmFactory.TypeOperator.parallel.name:
                    hmm = HmmFactory.factory(op1 * op2, self.n_states, self.n_samples)
                # Choice
                elif exp == HmmFactory.TypeOperator.choice.name:
                    hmm = HmmFactory.factory(op1 | op2, self.n_states, self.n_samples)
                # Add new hmm
                stack.append(hmm)

            # Unary operators
            elif exp in [HmmFactory.TypeOperator.iterative.name]:
                # Take operand
                op1 = stack.pop()
                # Iterative
                if(exp == HmmFactory.TypeOperator.iterative.name):
                    hmm = HmmFactory.factory(~op1, self.n_states, self.n_samples)
                # Add hmm
                stack.append(hmm)

            # Type Expression
            elif exp in [HmmFactory.TypeOperator.unistroke.name,
                         HmmFactory.TypeOperator.multistroke.name]:
                # Take operand
                op1 = stack.pop()
                # Take primitive expression
                if(exp == HmmFactory.TypeOperator.unistroke.name):
                    primitive = HmmFactory.factory(OneDollarModels.getModel(op1), self.n_states, self.n_samples)
                elif(exp == HmmFactory.TypeOperator.multistroke.name):
                    print(exp)
                    primitive = HmmFactory.factory(MDollarModels.getModel(op1), self.n_states, self.n_samples)
                # Add primitive expression
                stack.append(primitive)
            else:
                # Add exp expression
                stack.append(exp)

        return stack.pop()


# Take a gesture type and return its complete model
class OneDollarModels:

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
        if(type_gesture == OneDollarModels.TypeGesture.triangle.name):
            definition = Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4)
        # x
        elif(type_gesture == OneDollarModels.TypeGesture.x.name):
            definition = Point(0,0) + Line(3,-3) + Line(0,3) + Line(-3,-3)
        # rectangle
        elif(type_gesture == OneDollarModels.TypeGesture.rectangle.name):
            definition = Point(0,0) + Line(0,-3) + Line(4,0) + Line(0, 3) + Line(-4,0)
        # circle
        elif(type_gesture == OneDollarModels.TypeGesture.circle.name):
            definition = Point(0,0) + Arc(-3,-3, cw=False) + Arc(3,-3, cw=False) + Arc(3,3, cw=False) + Arc(-3,3, cw=False)
        # check
        elif(type_gesture == OneDollarModels.TypeGesture.check.name):
            definition = Point(0,0) + Line(2, -2) + Line(4,6)
        # caret
        elif(type_gesture == OneDollarModels.TypeGesture.caret.name):
            definition = Point(0,0) + Line(2,3) + Line(2,-3)
        # question mark
        elif(type_gesture == OneDollarModels.TypeGesture.question_mark.name):
            definition = Point(0,0) + Arc(2,2) + Arc(2,-2) + Arc(-2,-2) + Line(0,-3)
        # left square bracket
        elif(type_gesture == OneDollarModels.TypeGesture.left_sq_bracket.name):
            definition = Point(0,0) + Line(-2,0) + Line(0,-4) + Line(2,0)
        # right square bracket
        elif(type_gesture == OneDollarModels.TypeGesture.right_sq_bracket.name):
            definition = Point(0,0) + Line(2,0) + Line(0, -4)  + Line(-2, 0)
        # v
        elif(type_gesture == OneDollarModels.TypeGesture.v.name):
            definition = Point(0,0) + Line(2,-3) + Line(2,3)
        # delete_mark
        elif(type_gesture == OneDollarModels.TypeGesture.delete_mark.name):
            definition = Point(0,0) + Line(2, -3) + Line(-2,0) + Line(2,3)
        # left curly brace
        elif(type_gesture == OneDollarModels.TypeGesture.left_curly_brace.name):
            definition = Point(0,0) + Arc(-5,-5, cw=False) + Line(0,-6) + Arc(-3,-3)  + Arc(3,-3) + Line(0,-6) + Arc(5,-5,cw=False)
        # right curly brace
        elif(type_gesture == OneDollarModels.TypeGesture.right_curly_brace.name):
            definition = Point(0,0) + Arc(5,-5) + Line(0,-6) + Arc(3,-3, cw=False) + Arc(-3,-3, cw=False) + Line(0,-6) + Arc(-5,-5)
        # star
        elif(type_gesture == OneDollarModels.TypeGesture.star.name):
            definition = Point(0,0) + Line(2,5) + Line(2, -5) + Line(-5, 3) + Line(6,0) + Line(-5, -3)
        # pigtail
        elif(type_gesture == OneDollarModels.TypeGesture.pigtail.name):
            definition = Point(0,0) + Arc(3,3, cw=False) + Arc(-1,1, cw=False) + Arc(-1,-1, cw=False) + Arc(3, -3, cw=False)
        # arrow
        elif(type_gesture == OneDollarModels.TypeGesture.arrow.name):
            definition = Point(0,0) + Line(6,4) + Line(-4,0) + Line(5,1) + Line(-1, -4)
        # zig_zag
        #if(type_gesture == OneDollarModels.TypeGesture.zig_zag.name):
            #definition = None

        return definition

# Take a gesture type and return its complete model
class MDollarModels:

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
        if(type_gesture == MDollarModels.TypeGesture.arrowhead.name):
            definition = Point(0, 0) + Line(6, 0) + Point(4, 2) + Line(2, -2) + Line(-2, -2)
        # h
        elif(type_gesture == MDollarModels.TypeGesture.H.name):
            definition = Point(0, 4) + Line(0, -4) + Point(-1, 2) + Line(5, 0) + Point(4, 4) + Line(0, -4)
        # n
        elif(type_gesture == MDollarModels.TypeGesture.N.name):
            definition = Point(0, 4) + Line(0, -4) + Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(0, -4)
        # i
        elif(type_gesture == MDollarModels.TypeGesture.I.name):
            definition = Point(0, 4) + Line(4, 0) + Point(2, 4) + Line(0, -4) + Point(0, 0) + Line(4, 0)
        # p
        elif(type_gesture == MDollarModels.TypeGesture.P.name):
            definition = Point(0, 0) + Line(0, -4) + Point(0, 0) + Arc(1, -1, cw=True) + Arc(-1, -1, cw=True)
        # t
        elif(type_gesture == MDollarModels.TypeGesture.T.name):
            definition = Point(-2, 0) + Line(4, 0) + Point(0, 0) + Line(0, -4)
        # six point star
        elif(type_gesture == MDollarModels.TypeGesture.six_point_star.name):
            definition = Point(0, 0.5) + Line(2, 2) + Line(2, -2) + Line(-4, 0) + Point(0, 2) + Line(4, 0) + Line(-2, -2) + Line(-2, 2)
        # d
        elif(type_gesture == MDollarModels.TypeGesture.D.name):
            definition = Point(0, 0) + Line(0, 4) + Point(0, 4) + Arc(2, -2, cw=True) + Point(2, 2) + Arc(-2, -2, cw=True)
        # asterisk
        elif(type_gesture == MDollarModels.TypeGesture.asterisk.name):
            definition = Point(4,3) + Line(-4,-3) + Point(0,3) + Line (4,-3) + Point(2,4) + Line(0, -4)
        # exclamation_point
        elif(type_gesture == MDollarModels.TypeGesture.exclamation_point.name):
            definition = Point(0, 20) + Line(0, -19)+ Point(0, 1) + Line(0, -1)
        # null
        elif(type_gesture == MDollarModels.TypeGesture.null.name):
            definition = Point(0,0) + Arc(-3,-3, cw=False) + Arc(3,-3, cw=False) + Arc(3,3, cw=False) + \
                         Arc(-3,3, cw=False) + Point(4,1) + Line(-8, -8)
        # pitchfork
        elif(type_gesture == MDollarModels.TypeGesture.pitchfork.name):
            definition = Point(-2,4)+Arc(2,-2, cw=False) + Point(0,2)+Arc(2,2, cw=False) + Point(0,4)+Line(0,-4)
        # half note
        elif(type_gesture == MDollarModels.TypeGesture.half_note.name):
            definition = Point(0,0)+Line(0,-4) + Point(0,-4)+Arc(-1,-1, cw=False) + Point(-1,-5)+Arc(1,1, cw=False)
        # x
        elif(type_gesture == MDollarModels.TypeGesture.X.name):
            definition = Point(4,3) + Line(-4,-3) + Point(0,3) + Line (4,-3)

        return definition



