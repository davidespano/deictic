from enum import Enum
from model.gestureModel import Point, Line, Arc

class DatasetExpressions:

    class TypeDataset(Enum):
        unistroke_1dollar = 0
        multistroke_1dollar = 1
        unica = 2
        shrec = 3

    @staticmethod
    def returnExpressions(selected_dataset):
        if selected_dataset == DatasetExpressions.TypeDataset.unistroke_1dollar:
            return DatasetExpressions.__returnDollar1Unistroke()
        if selected_dataset == DatasetExpressions.TypeDataset.multistroke_1dollar:
            return DatasetExpressions.__returnDollar1Multistroke()
        if selected_dataset == DatasetExpressions.TypeDataset.unica:
            return DatasetExpressions.__returnUnica()
        if selected_dataset == DatasetExpressions.TypeDataset.shrec:
            return DatasetExpressions.__returnShrec()

    @staticmethod
    def __returnDollar1Unistroke():
        return {
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
            'delete mark': [
                Point(0, 0) + Line(2, -3) + Line(-2, 0) + Line(2, 3)
            ],
            'left curly brace': [
                Point(0, 0) + Arc(-5, -5, cw=False) + Arc(-3, -3) + Arc(3, -3) + Arc(5, -5, cw=False)
            ],
            'left square bracket': [
                Point(0, 0) + Line(-4, 0) + Line(0, -5) + Line(4, 0)
            ],
            'triangle':[
                Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4)
            ],
            'x':[
                Point(0,0) + Line(3,-3) + Line(0,3) + Line(-3,-3)
            ],
            'rectangle':[
                Point(0,0) + Line(0,-3) + Line(4,0) + Line(0, 3) + Line(-4,0)
            ],
            'question_mark':[
                Point(0,0) + Arc(4,4) + Arc(4,-4) + Arc(-4,-4) + Arc(-2,-2, cw=False) + Arc(2, -2, cw=False)
            ],
            'right square bracket':[
                Point(0,0) + Line(4,0) + Line(0, -5)  + Line(-4, 0)
            ],
            'v':[
                Point(0,0) + Line(2,-3) + Line(2,3)
            ],
            'right curly brace':[
                Point(0,0) + Arc(5,-5) +  Arc(3,-3, cw=False) + Arc(-3,-3, cw=False) + Arc(-5,-5)
            ],
            'star':[
                Point(0,0) + Line(2,5) + Line(2, -5) + Line(-5, 3) + Line(6,0) + Line(-5, -3)
            ],
            'pigtail':[
                Point(0,0) + Arc(3,3, cw=False) + Arc(-1,1, cw=False) + Arc(-1,-1, cw=False) + Arc(3, -3, cw=False)
            ],
        }

    @staticmethod
    def __returnDollar1Multistroke():
        return {
            'arrowhead': [
                (Point(0, 0) + Line(6, 0) + Point(4, 2) + Line(2, -2) + Line(-2, -2)),
                (Point(4, 2) + Line(2, -2) + Line(-2, -2) + Point(0, 0) + Line(6, 0)),
            ],
            'asterisk': [
                ((Point(4, 4) + Line(-4, -4)) + Point(0, 4) + Line(4, -4) + Point(2, 4) + Line(0, -4))
            ],
            'D': [
                # Point(0, 0) + Line(0, 6) + Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0),
                Point(0, 6) + Line(0, -6) + Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3,
                                                                                                 cw=True) + Line(-2, 0),
                # Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0) + Point(0, 0) + Line(0, 6),
                # Point(0, 6) + Line(2, 0) + Arc(3, -3, cw=True) + Arc(-3, -3, cw=True) + Line(-2, 0) + Point(0, 6) + Line(0, -6)
            ],
            'exclamation_point': [
                Point(0, 4) + Line(0, -3) + Point(0, 1) + Line(0, -1),
            ],
            'H': [
                (Point(0, 4) + Line(0, -4) + Point(0, 2) + Line(4, 0) + Point(4, 4) + Line(0, -4)),
                (Point(0, 4) + Line(0, -4) + Point(4, 4) + Line(0, -4) + Point(0, 2) + Line(4, 0)),
                # (Point(4, 4) + Line(0, -4) + Point(0, 4) + Line(0, -4) + Point(0, 2) + Line(4, 0))
            ],
            'half_note': [
                (Point(0, 0) + Arc(-3, 3, cw=False) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3,
                                                                                                         cw=False) + Point(
                    2, 16) + Line(0, -20)),
                (Point(2, 16) + Line(0, -20) + Point(0, 0) + Arc(-3, 3, cw=False) + Arc(-3, -3, cw=False) + Arc(3, -3,
                                                                                                                cw=False) + Arc(
                    3, 3, cw=False)),
                (Point(0, 0) + Arc(-3, 3, cw=False) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3,
                                                                                                         cw=False) + Point(
                    2, -4) + Line(0, 20)),
                (Point(2, -4) + Line(0, 20) + Point(0, 0) + Arc(-3, 3, cw=False) + Arc(-3, -3, cw=False) + Arc(3, -3,
                                                                                                               cw=False) + Arc(
                    3, 3, cw=False)),
            ],
            'T': [
                Point(-2, 0) + Line(4, 0) + Point(0, 0) + Line(0, -4),
                #Point(-2, 0) + Line(4, 0) + Point(-4, 2) + Line(0, 4),
                #Point(2, 0) + Line(-4, 0) + Point(0, 0) + Line(0, -4),
                #Point(2, 0) + Line(-4, 0) + Point(-4, 2) + Line(0, 4),
                Point(0, 0) + Line(0, -4) + Point(-2, 0) + Line(4, 0),
                #Point(0, 0) + Line(0, -4) + Point(2, 0) + Line(-4, 0),
                #Point(-4, 2) + Line(0, 4) + Point(-2, 0) + Line(4, 0),
                #Point(-4, 2) + Line(0, 4) + Point(2, 0) + Line(-4, 0)
            ],

            'N': [
                (Point(0, 4) + Line(0, -4) + Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(0, -4))
            ],

            'P': [
                #Point(0, 0) + Line(0, 8) + Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0),
                Point(0, 8) + Line(0, -8) + Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0),
                #Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0) + Point(0, 8) + Line(0, -8),
                #Point(0, 8) + Line(2, 0) + Arc(2, -2, cw=True) + Arc(-2, -2, cw=True) + Line(-2, 0) + Point(0, 0) + Line(0, 8)
            ],

            'X': [
                # (Point(0, 0) + Line(4, 4) + Point(4, 0) + Line(-4, 4)), NO
                (Point(0, 0) + Line(4, 4) + Point(0, 4) + Line(4, -4)),  # 33
                # (Point(4, 4) + Line(-4, -4) + Point(4, 0) + Line(-4, 4)), NO
                (Point(4, 4) + Line(-4, -4) + Point(0, 4) + Line(4, -4)),  # 174
                # (Point(4, 0) + Line(-4, 4) + Point(0, 0) + Line(4, 4)), # NO
                # (Point(4, 0) + Line(-4, 4) + Point(4, 4) + Line(-4, -4)), #NO
                (Point(0, 4) + Line(4, -4) + Point(0, 0) + Line(4, 4)), # 44
                (Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(-4, -4))  # 368
            ],
            'I': [
                (Point(0, 4) + Line(4, 0) + Point(2, 4) + Line(0, -4) + Point(0, 0) + Line(4, 0)),
                (Point(2, 4) + Line(0, -4) + Point(0, 0) + Line(4, 0) + Point(0, 4) + Line(4, 0)),
                (Point(0, 4) + Line(4, 0) + Point(0, 0) + Line(4, 0) + Point(2, 0) + Line(0, 4)),
                (Point(2, 4) + Line(0, -4) + Point(0, 4) + Line(4, 0) + Point(0, 0) + Line(4, 0))
            ],

            'null': [
                (Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False) + Point(4, 1) + Line(-8, -8)),  # 410
                (Point(0, 0) + Arc(-3, -3, cw=False) + Arc(3, -3, cw=False) + Arc(3, 3, cw=False) + Arc(-3, 3, cw=False) + Point(-4, -7) + Line(8, 8)),  # 118
            ],
            'pitchfork': [
                (Point(-2, 4) + Arc(2, -2, cw=False) + Arc(2, 2, cw=False) + Point(0, 4) + Line(0, -4)),
                (Point(0, 4) + Line(0, -4) + Point(-2, 4) + Arc(2, -2, cw=False) + Arc(2, 2, cw=False))
            ],

            'six_point_star': [
                (Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4) + Point(-2, -1) + Line(4, 0) + Line(-2, -4) + Line(-2,4)),
                (Point(-2, -1) + Line(4, 0) + Line(-2, -4) + Line(-2, 4) + Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2,4)),
                (Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0) + Point(-2, 1) + Line(4, 0) + Line(-2, -4) + Line(-2,4)),
                (Point(-2, 1) + Line(4, 0) + Line(-2, -4) + Line(-2, 4) + Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0)),
                (Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0) + Point(-2, 1) + Line(2, -4) + Line(2, 4) + Line(-4, 0)),
                (Point(-2, 1) + Line(2, -4) + Line(2, 4) + Line(-4, 0) + Point(-2, -2) + Line(2, 4) + Line(2, -4) + Line(-4, 0)),
                (Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4) + Point(-2, -1) + Line(2, -4) + Line(2, 4) + Line(-4, 0)),
                (Point(-2, -1) + Line(2, -4) + Line(2, 4) + Line(-4, 0) + Point(0, 0) + Line(-2, -4) + Line(4, 0) + Line(-2, 4))
            ]
        }

    @staticmethod
    def __returnUnica():
        return None

    @staticmethod
    def __returnShrec():
        return {

            'gesture_2': [
                Point(0, 0) + Line(-4, 2),
                Point(0, 0) + Line(-4, 2) + Line(4, -2),
                #Point(0, 0) + Line(0, -4) + Line(-4, 2),
                #Point(0, 0) + Line(2, -4) + Line(2, 4),
            ],

            'gesture_7': [
                Point(0, 0) + Line(4, 0),
                Point(0,0) + Line(4,0) + Line(-4,0)
            ],

            'gesture_8': [
                Point(4, 0) + Line(-4, 0),
                Point(4, 0) + Line(-4, 0) + Line(4, 0)
            ],

            'gesture_9': [
                Point(0, 0) + Line(0, 4),
                Point(0, 0) + Line(0, 4) + Line(0, -4)
            ],

            'gesture_10': [
                Point(0, 4) + Line(0, -4),
                Point(0, 4) + Line(0, -4) + Line(0, 4)
            ],

            'gesture_11': [
                Point(0, 4) + Line(4, -4) + Line(-4, 0) + Line(4,4),
                Point(4, 4) + Line(-4, -4) + Line(0, 4) + Line(4,-4)
            ],

            'gesture_12': [
                Point(0,4) + Line(0,-4) + Line(-2,2) + Line(4,0),
                Point(0,2) + Line(0,2) + Line(0,-4) + Line(-2,2) + Line(4,0),
                ],

            'gesture_13': [
                Point(2, 4) + Line(-2, -4) + Line(-2, 4),
                Point(-2, 4) + Line(2, -4) + Line(2, 4)
            ]
        }


'''
class __Parse:


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
'''