from dataset import Sequence
from model import Point, Line, Arc
import matplotlib.pyplot as plt
import math
import numpy


# ----------------------------- Create sequence from expressions ---------------------------#
class ArtificialSequence(Sequence):
    # features #
    __scale = 1

    def __init__(self, expressions=None, filename=None, spu=1):
        # check

        # sample per unit: spu / num_operands
        self.__spu = round(spu/ (len(expressions[0].get_operands())-1) )
        # create points
        self.expressions = expressions
        self.points = self.__createTrajectory()
        self.filename = filename
        self.compositeTransform = CompositeTransform()

    # public methods #
    def save(self, file_path):
        numpy.savetxt(file_path, self.points, delimiter=',')
    def plot(self):
        plt.plot(self.points[:,0], self.points[:,1], color='b')
        plt.scatter(self.points[:, 0], self.points[:, 1])
        for i in range(0, len(self.points)):
            plt.annotate(str(i), (self.points[i, 0], self.points[i, 1]))
        plt.axis('equal')
        plt.show()

    # private methods #
    def __createTrajectory(self):
        points = []
        startPoint = []

        for exp in self.expressions[0].get_operands():
            # Point
            if isinstance(exp, Point):
                points += [(exp.x,exp.y)]
            # Line
            if isinstance(exp, Line):
                points += ArtificialSequence.__createLine(startPoint, exp.dx, exp.dy, spu=self.__spu)
            # Arc
            if isinstance(exp, Arc):
                exp.dz = 0;
                distance = abs(0.5 * math.pi * exp.dx)
                num_states = round(distance * self.__spu + 0.5)
                points += ArtificialSequence.__createArc(startPoint, exp, num_states)

            startPoint=[points[-1][0]/ArtificialSequence.__scale, points[-1][1]/ArtificialSequence.__scale]
        return numpy.array(points)


    @staticmethod
    def __createLine(startPoint, dx, dy, spu=0):
        step_x = dx / max(spu - 1, 1)
        step_y = dy / max(spu - 1, 1)
        points = []
        for i in range(1, spu):
            a = (startPoint[0] + (i * step_x)) * ArtificialSequence.__scale
            b = (startPoint[1] + (i * step_y)) * ArtificialSequence.__scale
            points.append([a,b])
        return points
    @staticmethod
    def __createArc(startPoint, exp, num_states):
        step = 0.5 * math.pi / max(num_states - 1, 1)
        beta = 0
        alpha = 0
        # TODO this may be better coded
        if exp.cw:
            if exp.dy > 0:
                if exp.dx > 0:
                    alpha = 0
                else:
                    alpha = 0.5 * math.pi
            else:
                if exp.dx > 0:
                    alpha = 1.5 * math.pi
                else:
                    alpha = math.pi
        else:
            if exp.dy > 0:
                if exp.dx > 0:
                    alpha = 0.5 * math.pi
                else:
                    alpha = math.pi
            else:
                if exp.dx > 0:
                    alpha = 0
                else:
                    alpha = 1.5 * math.pi
        beta = alpha + math.pi
        points = []
        for i in range(1, num_states):
            a = (math.cos(beta) + math.cos(alpha)) * abs(exp.dx) + startPoint[0]
            b = (math.sin(beta) + math.sin(alpha)) * abs(exp.dy) + startPoint[1]
            points.append([a,b])
            if exp.cw:
                beta -= step
            else:
                beta += step
        return points