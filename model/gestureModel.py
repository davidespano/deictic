from abc import ABCMeta, abstractmethod
from enum import Enum
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy
from dataset import *


class OpEnum(Enum):
    Undef = -1
    Point = 0
    Line = 1
    Arc = 2
    Sequence = 3
    Choice = 4
    Disabling = 5
    Iterative = 6
    Parallel = 7
    Point3D = 8
    Line3D = 9

    @staticmethod
    def isGround(opEnum):
        return opEnum == OpEnum.Point or opEnum == OpEnum.Line or opEnum == OpEnum.Arc

class GestureExp:
    __metaclass__ = ABCMeta

    def __add__(self, other):
        return CompositeExp(self, other, OpEnum.Sequence)

    def __mul__(self, other):
        return CompositeExp(self, other, OpEnum.Parallel)

    def __or__(self, other):
        return CompositeExp(self, other, OpEnum.Choice)

    def __mod__(self, other):
        return CompositeExp(self, other, OpEnum.Disabling)

    def __invert__(self):
        return IterativeExp(self)




    def get_path(self, path, current):
        return None

    def get_points(self, points):
        return None

    def is_composite(self):
        return False


    def plot(self):
        pathList = list()
        self.get_path(pathList, Point(0, 0))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_xticks(numpy.arange(-50, 50, 1))
        ax.set_yticks(numpy.arange(-50, 50, 1))
        #codes, verts = zip(*pathList)
        #path = mpath.Path(verts, codes)
        #patch = patches.PathPatch(path, facecolor='None', lw=3)
        for patch in pathList:
            ax.add_patch(patch)
        ax.set_axisbelow(True)
        #x, y = zip(*path.vertices)
        #line, = ax.plot(x, y, 'go-')
        plt.axis('equal')
        plt.show()
        #return path

    def to_point_sequence(self):
        pointList = list()
        self.get_points(pointList)
        return numpy.array(pointList)



class CompositeExp(GestureExp):
    def __init__(self, left, right, op):
        self.parent = None
        self.left = left
        self.right = right
        self.right.parent = self
        self.left.parent = self
        self.op = op

    def __str__(self):
        op = "."
        if self.op == OpEnum.Sequence:
            op = "+"
        elif  self.op == OpEnum.Parallel:
            op = "*"
        elif self.op == OpEnum.Choice:
            op = "|"
        elif self.op == OpEnum.Disabling:
            op = '|='
        if self.parent is None or self.parent.op == self.op :
            return "{0} {1} {2}".format(str(self.left), op, str(self.right))
        else:
            return "({0} {1} {2})".format(str(self.left), op, str(self.right))

    def is_composite(self):
        return True

    def get_path(self, path, current):
        self.left.get_path(path, current)
        return self.right.get_path(path, current)

    def get_points(self, points):
        if not self.left is None:
            self.left.get_points(points)
        if not self.right is None:
            self.right.get_points(points)


class IterativeExp(GestureExp):
    def __init__(self, exp):
        self.exp = exp

    def is_composite(self):
        return True;

    def __str__(self):
        return "~{0}".format(str(self.exp))

    def get_path(self, path, current):
        if not self.exp is None:
            return self.exp.get_path(path, current)
        return None

    def get_points(self, points):
        if not self.exp is None:
            return self.exp.get_points(points)
        return None

################################### 2D ###################################
class Point(GestureExp):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "P({0},{1})".format(str(self.x), str(self.y))

    def get_path(self, path, current):
        path.append(patches.Ellipse(
            xy=(self.x, self.y),
            width=0.3, height=0.3, lw=3.0,
            edgecolor='black', facecolor='black'))
        current.x = self.x
        current.y = self.y
        return path

    def get_points(self, points):
        points.append([self.x, self.y, self])


class Line(GestureExp):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def __str__(self):
        return "l({0},{1})".format(str(self.dx), str(self.dy))

    def get_path(self, path, current):
        path.append(patches.FancyArrowPatch(
            (current.x, current.y),
            (current.x + self.dx, current.y + self.dy),
            arrowstyle='-|>',
            edgecolor='black', facecolor='black', mutation_scale = 20,
            lw=3.0))
        current.x += self.dx
        current.y += self.dy
        return path

    def get_points(self, points):
        last = points[-1]
        if last is not None:
            points.append([last[0] + self.dx, last[1] + self.dy, self])

class Arc(GestureExp):
    def __init__(self, dx, dy, cw=True):
        self.dx = dx
        self.dy = dy
        self.cw = cw

    def __str__(self):
        return "a({0},{1})".format(self.dx, self.dy)

    def get_path(self, path, current):
        if self.cw:
            path.append(patches.FancyArrowPatch(
                (current.x + self.dx, current.y + self.dy),
                (current.x, current.y),
                arrowstyle='<|-',
                edgecolor='black', facecolor='black', mutation_scale=20,
                lw=3.0,
                connectionstyle='arc3, rad=0.5'))

        else:
            path.append(patches.FancyArrowPatch(
                (current.x, current.y),
                (current.x + self.dx, current.y + self.dy),
                arrowstyle='-|>',
                edgecolor='black', facecolor='black', mutation_scale=20,
                lw=3.0,
                connectionstyle='arc3, rad=0.5'))
        current.x += self.dx
        current.y += self.dy
        return path

    def get_points(self, points):
        last = points[-1]
        if last is not None:
            points.append([last[0] + self.dx, last[1] + self.dy, self])


################################### 3D ###################################
class Point3D(GestureExp):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "l({0},{1},{2})".format(str(self.dx), str(self.dy), str(self.dz))

    def get_path(self, path, current):
        path.append(patches.Ellipse(
            xyz=(self.x, self.y, self.z),
            width=0.3, height=0.3, lw=3.0,
            edgecolor='black', facecolor='black'))
        current.x = self.x
        current.y = self.y
        current.z = self.z
        return path

    def get_points(self, points):
        points.append([self.x, self.y, self.z, self])

class Line3D(GestureExp):
    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def __str__(self):
        return "l({0},{1},{2})".format(str(self.dx), str(self.dy), str(self.dz))

    def get_path(self, path, current):
        path.append(patches.FancyArrowPatch(
            (current.x, current.y, current.z),
            (current.x + self.dx, current.y + self.dy, current.z + self.dz),
            arrowstyle='-|>',
            edgecolor='black', facecolor='black', mutation_scale = 20,
            lw=3.0))
        current.x += self.dx
        current.y += self.dy
        current.z += self.dz
        return path

    def get_points(self, points):
        last = points[-1]
        if last is not None:
            points.append([last[0] + self.dx, last[1] + self.dy, self, last[2]+self.dz])

    def plot(self):
        pathList = list()
        self.get_path(pathList, Point3D(0, 0, 0))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_xticks(numpy.arange(-50, 50, 1))
        ax.set_yticks(numpy.arange(-50, 50, 1))
        #codes, verts = zip(*pathList)
        #path = mpath.Path(verts, codes)
        #patch = patches.PathPatch(path, facecolor='None', lw=3)
        for patch in pathList:
            ax.add_patch(patch)
        ax.set_axisbelow(True)
        #x, y = zip(*path.vertices)
        #line, = ax.plot(x, y, 'go-')
        plt.axis('equal')
        plt.show()
        #return path