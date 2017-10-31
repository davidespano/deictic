from dataset.csvDataset import CsvDataset
import math
from pykalman import KalmanFilter

# Vector Class
class Vector():

    def __init__(self, x, y, z=None):
        """

        :param x:
        :param y:
        """
        # X
        if not isinstance(x, float):
            raise("the parameters must be float")
        else:
            self.x = x
        # Y
        if not isinstance(y, float):
            raise("the parameters must be float")
        else:
            self.y = y
        # Z
        if z != None and not isinstance(z, float):
            raise("the parameters must be float")
        else:
            self.z = z
    def __init__(self, values):
        if not isinstance(values, list) and len(list) >= 2 and len(list) <=3:
            raise("values must be a list which contains two or three elements")
        elif not isinstance(values[0], float):
            raise("values must contain float")
        else:
            self.x = values[0]
            self.y = values[1]
            if len(values) == 3:
                self.z = values[2]

    def __sub__(self, other):
        """

        :param other:
        :return:
        """
        if isinstance(other, Vector):
            if self.z == None:
                return math.hypot(other.x-self.x, other.y-self.y)
            else:
                return math.hypot(other.x-self.x, other.y-self.y, other.z-self.z)

# Extend CsvDataset with the class Vector
class CsvDataset(CsvDataset):

    # Override
    def readFile(self, filename, dim=2):
        # Get sequence
        points = super(CsvDataset, self).readFile(filename=filename)

        new_sequence = []
        # Convert sequence in a set of vectors
        for point in points:
            # Get coordinates
            values = []
            for index in range(0, len(dim)):
                values.append(point[index])
            new_sequence.append(Vector(values))

        return new_sequence

    # Override
    def readDataset(self, dim=2):
        # Get sequences
        filenames = super(CsvDataset, self).getDatasetIterator()

        new_sequences = []
        for filename in filenames:
            new_sequences.append(self.readFile(filename=filename))

        return new_sequences



"""
dataset = CsvDataset(dir)
for item in dataset.readDataset():
    sequence = item[0]
    filename = item[1]
    
    plt.plot(item[:,0],item[:,1])
    plt.title(filename)
    plt.show()
"""