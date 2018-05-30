from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy
from dataset import *
## Matplotlib
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, PathPatch
# register Axes3D class with matplotlib by importing Axes3D
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D
from shlex import shlex

class TypeRecognizer(Enum):
    online = 0
    offline = 1

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
    """

    """
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
        """

        :param path:
        :param current:
        :return:
        """
        return None

    def get_points(self, points):
        """

        :param points:
        :return:
        """
        return None

    def is_composite(self):
        """

        :return:
        """
        return False


    def plot(self):
        """

        :return:
        """
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

    def clone(self):
        return GestureExp()


class CompositeExp(GestureExp):
    def __init__(self, left, right, op):
        self.parent = None
        self.left = left
        self.right = right
        if self.right is not None:
            self.right.parent = self
        if self.left is not None:
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

    def clone(self):
        leftClone = None
        rightClone = None

        if self.left is not None:
            leftClone = self.left.clone()
        if self.right is not None:
            rightClone = self.right.clone()

        return CompositeExp(leftClone, rightClone, self.op)

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

    def get_operands(self):
        #todo: manage disabling and parallal case
        operands = []
        exp = self
        while isinstance(exp, CompositeExp):
            operands = [exp.right] + operands
            exp = exp.left
        return [exp]+operands

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

    def clone(self):
        cloneExp = None
        if self.exp is not None:
            cloneExp = self.exp.clone()
        return IterativeExp(cloneExp)

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

    def clone(self):
        return Point(self.x, self.y)


class Line(GestureExp):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def __str__(self):
        return "L({0},{1})".format(str(self.dx), str(self.dy))

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

    def clone(self):
        return Line(self.dx, self.dy)

class Arc(GestureExp):
    def __init__(self, dx, dy, cw=True):
        self.dx = dx
        self.dy = dy
        self.cw = cw

    def __str__(self):
        return "A({0},{1},{2})".format(self.dx, self.dy, self.cw)

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

    def clone(self):
        return Arc(self.dx, self.dy, self.cw)


class ParseEnum(Enum):
    Error = -1
    Exp = 0
    Point = 1
    Line = 2
    Arc = 3
    Arg1 = 4
    Arg2 = 5
    Arg3 = 6
    EndExp = 7


class StringParser:
    def __init__(self):
        self.state = ParseEnum.Exp

    def fromString(self, string):
        # TODO handle parenthesis in expressions
        stack = []
        parent = None
        current = None
        factor = 1
        tokens = list(shlex(string))
        for t in tokens:
            # START EXP
            if self.state == ParseEnum.Exp:
                if t == 'P':
                    current = Point(0, 0)
                    self.state = ParseEnum.Point
                elif t == 'L':
                    current = Line(0, 0)
                    self.state = ParseEnum.Line
                elif t == 'A':
                    current = Arc(0, 0, False)
                    self.state = ParseEnum.Arc
                elif (t == '+' or t == '*' or t == '|') and current is not None:
                    composite = CompositeExp(current, None, self.__getOp(t))
                    parent = composite
                    current = None
                    self.state = ParseEnum.Exp
                else:
                    self.state = ParseEnum.Error
            elif self.state == ParseEnum.Point or self.state == ParseEnum.Line or self.state == ParseEnum.Arc:
                if t == '(':
                    self.state = ParseEnum.Arg1
                else:
                    self.state = ParseEnum.Error
            # ARG 1
            elif self.state == ParseEnum.Arg1:
                if t == '-':
                    factor = -1
                elif t == ',':
                    self.state = ParseEnum.Arg2
                else:
                    try:
                        val = float(t)
                        if isinstance(current, Point):
                            current.x = val * factor
                        else:
                            current.dx = val * factor
                        factor = 1;

                    except ValueError:
                        self.state = ParseEnum.Error
            # ARG 2
            elif self.state == ParseEnum.Arg2:
                if t == '-':
                    factor = -1
                elif t == ',':
                    self.state = ParseEnum.Arg3
                else:
                    try:
                        val = float(t)
                        if isinstance(current, Point):
                            current.y = val * factor
                        else:
                            current.dy = val * factor
                        factor = 1
                        if not isinstance(current, Arc):
                            self.state = ParseEnum.EndExp

                    except ValueError:
                        self.state = ParseEnum.Error
            # ARG 3
            elif self.state == ParseEnum.Arg3:
                if t == 'false':
                    current.cw = False
                    self.state = ParseEnum.EndExp
                elif t == 'true':
                    current.cw = True
                    self.state = ParseEnum.EndExp
                else:
                    self.state = ParseEnum.Error
            # END EXP
            elif self.state == ParseEnum.EndExp:
                if t == ')':
                    if parent is not None:
                        parent.right = current
                        current = parent
                        parent = None
                    self.state = ParseEnum.Exp
                else:
                    self.state = ParseEnum.Error

            if self.state == ParseEnum.Error:
                return None
        return current

    def __getOp(self, t):
        if t == '+':
            return OpEnum.Sequence
        if t == '*':
            return OpEnum.Parallel
        if t == '|':
            return OpEnum.Choice














################################### 3D ###################################
class Point3D(GestureExp):

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "P3D({0},{1},{2})".format(str(self.x), str(self.y), str(self.z))

    def get_path(self, path, current):
        # Create a circle 2D object and transform it in 3D
        path.append(patches.Circle(
            xyz=(self.x, self.y),
            width=0.3, height=0.3, lw=3.0,
            edgecolor='black', facecolor='black'))
        # Update current position
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
        return "L3D({0},{1},{2})".format(str(self.dx), str(self.dy), str(self.dz))

    def get_path(self, path, current):
        # Create an arrow patch 2D and transform it in 3D
        path.append(patches.FancyArrowPatch(
            (current.x, current.y),
            (current.x + self.dx, current.y + self.dy),
            arrowstyle='-|>',
            edgecolor='black', facecolor='black', mutation_scale=20,
            lw=3.0))
        # Udpate current position
        current.x += self.dx
        current.y += self.dy
        current.z += self.dz
        return path

    def get_points(self, points):
        last = points[-1]
        if last is not None:
            points.append([last[0]+self.dx, last[1]+self.dy, last[2]+self.dz, self])

    def plot(self):
        # Create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Get patches and add them on a 3D plot
        pathList = list()
        # Add origin
        self.get_path(pathList, Point3D(0, 0, 0))
        # Add patches
        # TODO fix zdir
        for patch in pathList:
            ax.add_patch(patch)
            art3d.pathpatch_2d_to_3d(patch, z=self.dz, zdir="y")
        # Plot settings
        ax.grid(True)
        ax.set_axisbelow(True)
        # Axis limit
        ax.set_xlim(-self.dx-10, self.dx+10)
        ax.set_ylim(-self.dy-10, self.dy+10)
        ax.set_zlim(-self.dz-10, self.dz+10)

        plt.show()
