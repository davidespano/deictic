from abc import ABCMeta, abstractmethod
from enum import Enum
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy
from dataset import *


class OpEnum(Enum):
    Sequence = 0
    Parallel = 1
    Choice = 2

class GestureExp:
    __metaclass__ = ABCMeta

    def __add__(self, other):
        return CompositeExp(self, other, OpEnum.Sequence)

    def __mul__(self, other):
        return CompositeExp(self, other, OpEnum.Parallel)

    def __or__(self, other):
        return CompositeExp(self, other, OpEnum.Choice)

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
        codes, verts = zip(*pathList)
        path = mpath.Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='None', lw=2)
        ax.add_patch(patch)
        x, y = zip(*path.vertices)
        line, = ax.plot(x, y, 'go-')
        plt.axis('equal')
        plt.show()
        return path

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
        elif self.op == OpEnum.Parallel:
            op = "|"
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



class Point(GestureExp):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "P({0},{1})".format(str(self.x), str(self.y))

    def get_path(self, path, current):
        current.x = self.x
        current.y = self.y
        return path.append((mpath.Path.MOVETO, (self.x, self.y)))

    def get_points(self, points):
        points.append([self.x, self.y, self])


class Line(GestureExp):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def __str__(self):
        return "l({0},{1})".format(str(self.dx), str(self.dy))

    def get_path(self, path, current):
        current.x += self.dx
        current.y += self.dy
        return path.append((mpath.Path.LINETO, (current.x, current.y)))

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
            if self.dx * self.dy <= 0:
                current.x += self.dx
                path.append((mpath.Path.CURVE3, (current.x, current.y)))
                current.y += self.dy
            else:
                current.y += self.dy
                path.append((mpath.Path.CURVE3, (current.x, current.y)))
                current.x += self.dx
        else:
            if self.dx * self.dy <= 0:
                current.y += self.dy
                path.append((mpath.Path.CURVE3, (current.x, current.y)))
                current.x += self.dx
            else:
                current.x += self.dx
                path.append((mpath.Path.CURVE3, (current.x, current.y)))
                current.y += self.dy
        return path.append((mpath.Path.CURVE3, (current.x, current.y)))

    def get_points(self, points):
        last = points[-1]
        if last is not None:
            points.append([last[0] + self.dx, last[1] + self.dy, self])

class ModelPreprocessor:

    def __init__(self, exp):
        self.exp = exp
        self.transforms = CompositeTransform()

    def preprocess(self):
        points = self.exp.to_point_sequence()
        transformed = self.transforms.transform(points)

        x = 0
        y = 0
        # update the expression terms
        for i in range(0,len(points)):
            if isinstance(points[i][2], Point):
                points[i][2].x = transformed[i][0] - x
                points[i][2].y = transformed[i][1] - y
                x = transformed[i][0]
                y = transformed[i][1]

            elif isinstance(points[i][2], Line):
                points[i][2].dx = transformed[i][0] - x
                points[i][2].dy = transformed[i][1] - y
                x = transformed[i][0]
                y = transformed[i][1]

            elif isinstance(points[i][2], Arc):
                points[i][2].dx = transformed[i][0] - x
                points[i][2].dy = transformed[i][1] - y
                x = transformed[i][0]
                y = transformed[i][1]




gesture_models = [
    #Point(0,0) + Line(-2,-3) + Line(4,0)+ Line(-2,3), # triangle
    #Point(0,0) + Line(3,-3) + Line(0,3) + Line(-3,-3), # X
    #Point(0,0) + Line(0,-3) + Line(4,0) + Line(0, 3) + Line(-4,0), # rectangle
    #Point(0,0) + Arc(-3,-3, cw=False) + Arc(3,-3, cw=False) + Arc(3,3, cw=False) + Arc(-3,3, cw=False), # circle
    #Point(0,0) + Line(2, -2) + Line(4,6), # check
    #Point(0,0) + Line(2,3) + Line(2,-3), # caret
    #Point(0,0) + Arc(2,2) + Arc(2,-2) + Arc(-2,-2) + Line(0,-3), # question mark
    #Point(0,0) + Line(6,4) + Line(-3,0) + Line(4,1) + Line(-1, -3), # arrow
    #Point(0,0) + Line(-2,0) + Line(0,-4) + Line(2,0), # left square bracket
    #Point(0,0) + Line(2,0) + Line(0, -4)  + Line(-2, 0), # right square bracket
    #Point(0,0) + Line(2,-3) + Line(2,3), # V
    #Point(0,0) + Line(2, -3) + Line(-2,0) + Line(2,3), # delete
    #Point(0,0) + Arc(-2,-2, cw=False) + Line(0,-3) + Arc(-1,-1) + Arc(1,-1) + Line(0,-3) + Arc(2,-2,cw=False), # left curly brace
    #Point(0,0) + Arc(2,-2) + Line(0,-3) + Arc(1,-1, cw=False) + Arc(-1,-1, cw=False) + Line(0,-3) + Arc(-2,-2),  # right curly brace
    #Point(0,0) + Line(2,5) + Line(2, -5) + Line(-5, 3) + Line(6,0) + Line(-5, -3), # star
    Point(0,0) + Arc(6,6, cw=False) + Arc(-1,1, cw=False) + Arc(-1,-1, cw=False) + Arc(6, -6, cw=False) # pigtail
]




for gesture in gesture_models:
    processor = ModelPreprocessor(gesture)
    transform1 = CenteringTransform()
    transform2 = NormaliseLengthTransform(axisMode=True)
    transform3 = ScaleDatasetTransform(scale=100)
    processor.transforms.addTranform(transform1)
    processor.transforms.addTranform(transform2)
    processor.transforms.addTranform(transform3)
    print(gesture)
    processor.preprocess()
    gesture.plot()