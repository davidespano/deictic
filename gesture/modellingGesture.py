from model import *

baseDir = '/home/alessandro/PycharmProjects/deictic/repository/'
trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'
testDir = baseDir + "deictic/mdollar-dataset/resampled/"

class GestureFactory:
    @staticmethod
    def factory(gesture):
        factory = ClassifierFactory()
        factory.setLineSamplesPath(trainingDir)
        factory.setClockwiseArcSamplesPath(arcClockWiseDir)
        factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
        factory.states = n_states
        factory.spu = n_samples
        model, edges = factory.createClassifier(gesture)
        return  model

# Take a gesture type and return its complete model
class OneDollarModels:

    class TypeGesture(Enum):
        triangle = 0
        X = 1
        Rectangle = 2
        Circle = 3
        Check = 4
        Caret = 5
        QuestionMark = 6
        LeftSquareBracket = 7
        RightSquareBracket = 8
        V = 9
        Delete = 10
        LeftCurlyBrace = 11
        RightCurlyBrace = 12
        Star = 13
        Pigtail = 14

    # triangle
    @staticmethod
    def triangle():
        return GestureFactory.factory(Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4))
    # x
    @staticmethod
    def x():
        return GestureFactory.factory(Point(0,0) + Line(3,-3) + Line(0,3) + Line(-3,-3))
    # rectangle
    @staticmethod
    def rectangle():
        return GestureFactory.factory(Point(0,0) + Line(0,-3) + Line(4,0) + Line(0, 3) + Line(-4,0))
    # circle
    @staticmethod
    def circle():
        return GestureFactory.factory(Point(0,0) + Arc(-3,-3, cw=False) + Arc(3,-3, cw=False) + Arc(3,3, cw=False) + Arc(-3,3, cw=False))
    # check
    @staticmethod
    def check():
        return GestureFactory.factory(Point(0,0) + Line(2,3) + Line(2,-3))
    # caret
    @staticmethod
    def caret():
        return GestureFactory.factory(Point(0,0) + Arc(2,2) + Arc(2,-2) + Arc(-2,-2) + Line(0,-3))
    # question mark
    @staticmethod
    def question_mark():
        return GestureFactory.factory(Point(0,0) + Line(6,4) + Line(-4,0) + Line(5,1) + Line(-1, -4))
    # left square bracket
    @staticmethod
    def left_square_bracket():
        return GestureFactory.factory(Point(0,0) + Line(-2,0) + Line(0,-4) + Line(2,0))
    # right square bracket
    @staticmethod
    def right_square_bracket():
        return GestureFactory.factory(Point(0,0) + Line(2,0) + Line(0, -4)  + Line(-2, 0))
    # v
    @staticmethod
    def v():
        return GestureFactory.factory(Point(0,0) + Line(2,-3) + Line(2,3))
    # delete
    @staticmethod
    def delete():
        return GestureFactory.factory(Point(0,0) + Line(2, -3) + Line(-2,0) + Line(2,3))
    # left curly brace
    @staticmethod
    def left_curly_brace():
        return GestureFactory.factory(Point(0,0) + Arc(-5,-5, cw=False) + Line(0,-6) + Arc(-3,-3)  + Arc(3,-3) + Line(0,-6) + Arc(5,-5,cw=False))
    # right curly brace
    @staticmethod
    def right_curly_brace():
        return GestureFactory.factory(Point(0,0) + Arc(5,-5) + Line(0,-6) + Arc(3,-3, cw=False) + Arc(-3,-3, cw=False) + Line(0,-6) + Arc(-5,-5))
    # star
    @staticmethod
    def star():
        return GestureFactory.factory(Point(0,0) + Line(2,5) + Line(2, -5) + Line(-5, 3) + Line(6,0) + Line(-5, -3))
    # pigtail
    @staticmethod
    def pigtail():
        return GestureFactory.factory(Point(0,0) + Arc(3,3, cw=False) + Arc(-1,1, cw=False) + Arc(-1,-1, cw=False) + Arc(3, -3, cw=False))

# Take a gesture type and return its complete model
class MDollarModels:

    class TypeGesture(Enum):
        Arrowhead = 0
        H = 1
        N = 2
        I = 3
        P = 4
        T = 5
        SixPointStar = 6
        D = 7
        Asterisk = 8
        ExclamationPoint = 9
        Null = 10
        Pitchfork = 11
        half_note = 12
        X = 13

    @staticMethod
    def getModel():

    # arrowhead
    def arrowhead():
        return GestureFactory.factory(Point(0, 0) + Line(6, 0) + Point(4, 2) + Line(2, -2) + Line(-2, -2))
    # h
    def h():
        return GestureFactory.factory(Point(0, 4) + Line(0, -4) + Point(-1, 2) + Line(5, 0) + Point(4, 4) + Line(0, -4))
    # n
    def n():
        return GestureFactory.factory(Point(0, 4) + Line(0, -4) + Point(0, 4) + Line(4, -4) + Point(4, 4) + Line(0, -4))
    # i
    def i():
        return GestureFactory.factory(Point(0, 4) + Line(4, 0) + Point(2, 4) + Line(0, -4) + Point(0, 0) + Line(4, 0))
    # p
    def p():
        return GestureFactory.factory(Point(0, 0) + Line(0, -4) + Point(0, 0) + Arc(1, -1, cw=True) + Arc(-1, -1, cw=True))
    # t
    def t():
        return GestureFactory.factory(Point(-2, 0) + Line(4, 0) + Point(0, 0) + Line(0, -4))
    # six point star
    def six_point_star():
        return GestureFactory.factory(Point(0, 0.5) + Line(2, 2) + Line(2, -2) + Line(-4, 0) + Point(0, 2) + Line(4, 0) + Line(-2, -2) + Line(-2, 2))
    # d
    def d():
        return GestureFactory.factory(Point(0, 0) + Line(0, 4) + Point(0, 4) + Arc(2, -2, cw=True) + Point(2, 2) + Arc(-2, -2, cw=True))
    # asterisk
    def asterisk():
        return GestureFactory.factory(Point(4,3) + Line(-4,-3) + Point(0,3) + Line (4,-3) + Point(2,4) + Line(0, -4))
    # exclamation_point
    def exclamation_point():
        return GestureFactory.factory(Point(0, 20) + Line(0, -19)+ Point(0, 1) + Line(0, -1))
    # null
    def null():
        return GestureFactory.factory(Point(0,0) + Arc(-3,-3, cw=False) + Arc(3,-3, cw=False) + Arc(3,3, cw=False) +
                                      Arc(-3,3, cw=False) + Point(4,1) + Line(-8, -8))
    # pitchfork
    def pitchfork():
        return GestureFactory.factory(Point(-2,4)+Arc(2,-2, cw=False) + Point(0,2)+Arc(2,2, cw=False) + Point(0,4)+Line(0,-4))
    # half note
    def half_note():
        return  GestureFactory.factory(Point(0,0)+Line(0,-4) + Point(0,-4)+Arc(-1,-1, cw=False) + Point(-1,-5)+Arc(1,1, cw=False))
    # x
    def x():
        return GestureFactory.factory(Point(4,3) + Line(-4,-3) + Point(0,3) + Line (4,-3))



