# Imports
import numpy


class Vector():
    def __init__(self, x, y):
        """

        :param x:
        :param y:
        """
        self.x = x
        self.y = y

    def __mul__(self, other):
        """

        :param vector_b:
        :return:
        """
        if(isinstance(other, Vector)):
            x = self.x * other.x
            y = self.y * other.y
            return Vector(x,y)
        else:
            raise ("vector_b must must be a Vector object.")


    def __sub__(self, other):
        """

        :param vector_b:
        :return:
        """
        if(isinstance(other, Vector)):
            x = self.x - other.x
            y = self.y - other.y
            return Vector(x,y)
        else:
            raise ("vector_b must must be a Vector object.")

    def norm(self):
        array = numpy.ndarray([self.x, self.y])
        return numpy.linalg.norm(array)

#
class Parsing():

    def parsingLine(self, sequence):
        # Kalmar smoother and threshold
        smoothed_sequence,threshold = self.kalmarSmoother()
        # Parsing line
        for t in range(1, len(sequence), 1):
            den = (sequence[t] - sequence[t-1]) * (sequence[t+1] - sequence[t])
            num = (sequence[t] - sequence[t-1]).norm() * (sequence[t+1]-sequence[t]).mul()
            delta = 1 - (den) / (num)

    def kalmarSmoother(self, sequence):
        threshold = 0

        return sequence, threshold

    def differencePoints(self, vector_a, vector_b):
        if len(vector_a) == len(vector_b):
            for index in range(0, len(vector_a)):
                x = vector_a.x - vector_b.x
                y = vector_a.y - vector_b.y
            return Vector(x,y)
        else:
            raise("vector_a and vector_b must have the same lenght.")

t_0 = Vector(0,0)
t_1 = Vector(1,1)
t_2 = Vector(2,2)
den = (t_1 - t_0).norm() * (t_2 - t_1).norm()
print(den)
num = (t_1 - t_0).norm() * (t_2 - t_1).norm()
print(num)

t_0 = numpy.ndarray([0,0])
t_1 = numpy.ndarray([1,1])
t_2 = numpy.ndarray([2,2])
den = t_1 - t_0
print(den)
