from enum import Enum
# Kalman filter
from pykalman import KalmanFilter
# Math
import math
# Numpy
import numpy as np
import numpy.linalg as la
# Operator
import operator
# Plot
import matplotlib.pyplot as plt
import dataset as dt

class MathUtils():

    class Directions(Enum):
        East = 0
        NorthEast = 1
        North = 2
        NorthWest = 3
        West = 4
        SouthWest = 5
        South = 6
        SouthEast = 7

    angle_directions = np.array([x for x in range(0, 360, 45)])

    __directionsVect = {
        Directions.North:[0,1],
        Directions.South:[0,-1],
        Directions.West:[-1,0],
        Directions.East:[1,0],
        Directions.NorthEast:[math.sqrt(2)/2, math.sqrt(2)/2],
        Directions.NorthWest:[-math.sqrt(2)/2, math.sqrt(2)/2],
        Directions.SouthEast:[math.sqrt(2)/2, -math.sqrt(2)/2],
        Directions.SouthWest:[-math.sqrt(2)/2, -math.sqrt(2)/2]
    }

    # todo: check parameters
    @staticmethod
    def findWay(points):
        """

        :param points:
        :return:
        """
        if MathUtils.areaPolygon(points) > 0:
            return True # Clockwise
        else:
            return False # Anticlockwise
    @staticmethod
    def areaPolygon(points):
        """
            -Shoelace formula-
        :param points:
        :return: bool
        """
        sum = 0
        for index in range(len(points)):
            sum += (points[index][0]*points[(index+1)%len(points)][1]) - points[(index+1)%len(points)][0]*points[index][1]
        return sum
    @staticmethod
    def findDirection(point):
        north = MathUtils.dot(point, MathUtils.__directionsVect[MathUtils.Directions.North])
        north_east = MathUtils.dot(point, MathUtils.__directionsVect[MathUtils.Directions.NorthEast])
        north_west = MathUtils.dot(point, MathUtils.__directionsVect[MathUtils.Directions.NorthWest])
        south = MathUtils.dot(point, MathUtils.__directionsVect[MathUtils.Directions.South])
        south_east = MathUtils.dot(point, MathUtils.__directionsVect[MathUtils.Directions.SouthEast])
        south_west = MathUtils.dot(point, MathUtils.__directionsVect[MathUtils.Directions.SouthWest])
        east = MathUtils.dot(point, MathUtils.__directionsVect[MathUtils.Directions.East])
        west = MathUtils.dot(point, MathUtils.__directionsVect[MathUtils.Directions.West])
        # North
        if north > south and north > east and north > west:
            if north > north_west and north > north_east:
                return MathUtils.Directions.North.value
            if north_west >= north and north_west > north_east:
                return MathUtils.Directions.NorthWest.value
            if north_east >= north and north_east > north_west:
                return MathUtils.Directions.NorthEast.value
        # South
        if south > north and south > east and south > west:
            if south > south_west and south > south_east:
                return MathUtils.Directions.South.value
            if south_west >= south and south_west > south_east:
                return MathUtils.Directions.SouthWest.value
            if south_east >= south and south_east > south_west:
                return MathUtils.Directions.SouthEast.value
        # East
        if east > north and east > south and east > west:
            if east > north_east and east > south_east:
                return MathUtils.Directions.East.value
            if north_east >= east and north_east > south_east:
                return MathUtils.Directions.NorthEast.value
            if south_east >= east and south_east > north_east:
                return MathUtils.Directions.SouthEast.value
        # West
        if west > north and west > south and west > east:
            if west > south_west and west > north_west:
                return MathUtils.Directions.West.value
            if south_west >= west and south_west > north_west:
                return MathUtils.Directions.SouthWest.value
            if north_west >= west and north_west > south_west:
                return MathUtils.Directions.NorthWest.value
        # error
        return None
    @staticmethod
    def findNearest(array, value):
        idx = (np.abs(array - value).argmin())
        return idx
    @staticmethod
    def surfaceTriangle(side_a, side_b, side_c):
        return (side_a+side_b+side_c)/2
    @staticmethod
    def curvature(side_a, side_b, side_c):
        surface = MathUtils.surfaceTriangle(side_a, side_b, side_c)
        try:
            val = (4 * math.sqrt(surface * (surface - side_a) * (surface - side_b) * (surface - side_c))) \
                / (side_a * side_b * side_c)
        except:
            val = 1
        return val
    @staticmethod
    def magn(point):
        return math.sqrt(math.pow(point[0],2)+math.pow(point[1],2))
    @staticmethod
    def dot(point_a, point_b):
        return point_a[0]*point_b[0] + point_a[1]*point_b[1]
    @staticmethod
    def sub(point_a, point_b):
        vector = [point_a[0]-point_b[0], point_a[1]-point_b[1]]
        return vector
    @staticmethod
    def distance(point_a, point_b):
        return math.hypot(point_b[0]-point_a[0], point_b[1]-point_a[1])
    @staticmethod
    def normalize(point):
        length = MathUtils.magn(point)
        for index in range(0, len(point)):
            point[index] = point[index]/length
        return point





class Trajectory():

    class TypePrimitive(Enum):
        NONE = "0"
        LINE = "A"
        ARC = "B" # intervals
        BOUNDARY = "O"

    def __init__(self, sequence):
        """

        :param sequence:
        """
        # Check parameters
        if not isinstance(sequence, np.ndarray):
            raise Exception("sequence must be a numpy array.")
        # Inizialization
        #
        self.__sequence = sequence
        #
        self.__labels = [Trajectory.TypePrimitive.NONE.value for x in range(len(sequence))]
        #
        self.__descriptors = np.zeros((len(sequence)), dtype=float)
        #
        self.__curvatures = [None for x in range(len(sequence))]

    # Methods #
    def getLabelsSequence(self):
        return self.__labels
    def getPointsSequence(self):
        return self.__sequence

    def algorithm1(self, threshold_a):
        """
            algorithm1 labels the sequence's points in accordance with the Algorithm 1 proposed in "Parsing 3D motion trajectory for gesture recognition".
        :param threshold_a:
        :return: list of label
        """
        # Parsing line
        for t in range(1, len(self.__sequence)-1):
            # [davide] inizio vecchio labelling
            # Compute delta
            #num1 = MathUtils.sub(self.__sequence[t], self.__sequence[t - 1])
            #num2 = MathUtils.sub(self.__sequence[t + 1], self.__sequence[t])
            #num = MathUtils.dot(num1, num2)
            #den1 = MathUtils.magn(MathUtils.sub(self.__sequence[t], self.__sequence[t - 1]))
            #den2 = MathUtils.magn(MathUtils.sub(self.__sequence[t + 1], self.__sequence[t]))
            #den = den1 * den2
            #delta = 1 - (num/den)
            # Check delta
            #if delta < threshold_a:
            #    self.__labels[t] = (Trajectory.TypePrimitive.LINE.value)#str(delta)
            # [davide] fine vecchio labelling
            if dt.Geometry2D.Collinear(
                    dt.Point2D(self.__sequence[t][0], self.__sequence[t][1]),
                    dt.Point2D(self.__sequence[t - 1][0], self.__sequence[t - 1][1]),
                    dt.Point2D(self.__sequence[t + 1][0], self.__sequence[t + 1][1]),
                    threshold_a
            ):
                self.__labels[t] = (Trajectory.TypePrimitive.LINE.value)  # str(delta)
        return self.__labels

    def algorithm2(self, threshold_b = None):
        """
            algorithm2 labels the sequence's points in accordance with the Algorithm 2 proposed in "Parsing 3D motion trajectory for gesture recognition".
        :param threshold_b:
        :return:
        """
        for t in range(0, len(self.__sequence)-1):
            if self.__labels[t] == Trajectory.TypePrimitive.NONE.value and self.__labels[t+1] == Trajectory.TypePrimitive.NONE.value:
                # compute volume
                # assigned label
                self.__labels[t] = Trajectory.TypePrimitive.ARC.value
                self.__labels[t + 1] = Trajectory.TypePrimitive.ARC.value
        return self.__labels

    def algorithm3(self):
        """
            algorithm3 provides to label the sequence's points, in accordance with the Algorithm 3 proposed in "Parsing 3D motion trajectory for gesture recognition"
            in order to localize the boundary points (isolated points or label transition).
        :param
        :return:
        """
        for t in range(0, len(self.__sequence)-1):
            if self.__labels[t] == Trajectory.TypePrimitive.NONE.value:
                self.__labels[t] = Trajectory.TypePrimitive.BOUNDARY.value # isolated points
            elif self.__labels[t+1] != Trajectory.TypePrimitive.NONE.value and self.__labels[t] != self.__labels[t+1]:
                self.__labels[t] = Trajectory.TypePrimitive.BOUNDARY.value # for label transition
        return self.__labels

    def descriptorTrajectory(self):
        """

        :return:
        """
        #
        for t in range(1, len(self.__sequence)-2):
            if self.__labels[t] == Trajectory.TypePrimitive.LINE.value:
                self.__descriptors[t] = 1
            elif self.__labels[t] == Trajectory.TypePrimitive.ARC.value:
                _lambda = 1
                ### get descriptor ###
                # curvature
                k = self.__computeCurvature(t)
                # approssimation curvature
                k_s = self.__approximationCurvature(t)
                # descriptor
                self.__descriptors[t] = math.sqrt(math.pow(k, 2) + _lambda * math.pow(k_s, 2))
        return self.__descriptors

    def findSubPrimitives(self, beta):
        """

        :param beta:
        :return:
        """
        indexes=[]
        for index in range(len(self.__sequence)-1):
            if self.__labels[index] == Trajectory.TypePrimitive.LINE.value:
                self.__quantizationIntervalsLine(indexes=[index])
                indexes.clear()
            elif self.__labels[index] == Trajectory.TypePrimitive.ARC.value:
                indexes.append(index)
                self.__quantizationIntervalsArc(indexes=indexes)
        return self.__labels

        # indexes.append(0)
        # for index in range(1, len(self.__sequence)):
        #     if self.__labels[index] != self.__labels[index - 1] or index == (len(self.__sequence)):
        #         if self.__labels[index-1] == Trajectory.TypePrimitive.ARC.value:
        #             self.__quantizationIntervalsArc(indexes=indexes, beta=beta)
        #         elif self.__labels[index-1] == Trajectory.TypePrimitive.LINE.value:
        #             self.__quantizationIntervalsLine(indexes=indexes)
        #         indexes.clear()
        #     indexes.append(index)
        # return self.__labels

    # private methods #
    def __computeCurvature(self, index):
        """

        :param index:
        :return:
        """
        if self.__curvatures[index] == None:
            point_b = self.__sequence[index - 1]
            point_t = self.__sequence[index]
            point_c = self.__sequence[index + 1]
            side_a = MathUtils.distance(point_a=point_b, point_b=point_t)
            side_b = MathUtils.distance(point_a=point_t, point_b=point_c)
            side_c = MathUtils.distance(point_a=point_b, point_b=point_c)
            self.__curvatures[index] = MathUtils.curvature(side_a, side_b, side_c)
        return self.__curvatures[index]

    def __approximationCurvature(self, index):
        """

        :param index:
        :return:
        """
        point_a = self.__sequence[index - 2]
        point_b = self.__sequence[index - 1]
        point_t = self.__sequence[index]
        point_c = self.__sequence[index + 1]
        point_d = self.__sequence[index + 2]
        side_a = MathUtils.distance(point_a=point_b, point_b=point_t)
        side_b = MathUtils.distance(point_a=point_t, point_b=point_c)
        side_d = MathUtils.distance(point_a=point_a, point_b=point_t)
        side_g = MathUtils.distance(point_a=point_c, point_b=point_d)
        curvature_prec = self.__computeCurvature(index - 1)
        curvature_next = self.__computeCurvature(index + 1)
        return 3 * ((curvature_prec - curvature_next) / (
                              2 * side_a + 2 * side_b + side_d + side_g))

    def __quantizationIntervalsArc(self, indexes=[], beta=1):
        """

        :param beta:
        :return:
        """
        descriptors = []
        for index in indexes:
            descriptors.append(self.__descriptors[index])
        # Get min and max value from its points
        min_value = min(descriptors)
        max_value = max(descriptors)
        # comput interval values
        interval_value = (max_value-min_value)/beta
        interval_values = [min_value+(interval_value*x) for x in range(beta)]
        for index in indexes:
            # assign interval
            interval_index = MathUtils.findNearest(interval_values, self.__descriptors[index])
            # find direction clockwise or counter-clockwise
            interval_direction = MathUtils.findWay([item for item in operator.itemgetter([index-1, index, index+1])(self.__sequence)])
            self.__labels[index] = Trajectory.TypePrimitive.ARC.value +str(interval_index)+str(interval_direction) #chr(ord(self.__labels[index])+interval_index+17)
    def __quantizationIntervalsLine(self, indexes=[]):
        """

        :param indexes:
        :param beta:
        :return:
        """
        # Compute angle
        # start_point = self.__sequence[indexes[0]-1]
        # end_point = self.__sequence[indexes[-1]]
        # point_direction = MathUtils.sub(end_point, start_point)
        # direction = MathUtils.findDirection(MathUtils.normalize(point_direction))
        # Get points
        for index in indexes:
            start_point = self.__sequence[index-1]
            end_point = self.__sequence[index]
            point_direction = MathUtils.sub(end_point, start_point)
            direction = MathUtils.findDirection(MathUtils.normalize(point_direction))
            self.__labels[index] = self.__labels[index]+str(direction) #chr(ord(self.__labels[index]) + interval_direction + 4)      #








'''
    This class implements the class necessary to parse a trajectory into two primitives (straight line or plane arc) by labelling each frame. 
'''
class Parsing():

    # Methods #
    @staticmethod
    def parsingLine(sequence, flag_plot=False, flag_save=False, path = None):
        """
            parsingLine provides to: apply a kalmar smoother to the sequence and label it in accordance with "Parsing 3D motion trajectory for gesture recognition"
        :param sequence: the sequence to be parsed.
        :param flag_plot: does the user want to plot the obtained sequence?
        :param flag_save: does the user want to save the obtained sequence?
        :return:
        """
        # Check parameters
        if not isinstance(sequence, np.ndarray):
            raise Exception("sequence must be a numpy ndarray.")

        threshold_a = 100#0.0025
        # trajectory
        trajectory = Trajectory(sequence)
        # Algorithm 1 (find straight linear)
        list = trajectory.algorithm1(threshold_a=threshold_a)
        # Algorithm 2 (find plane arc)
        #list = trajectory.algorithm2(threshold_b=None)
        # Algorithm 3 (localizing boundary points)
        #list = trajectory.algorithm3()
        # Descriptor
        #f = trajectory.descriptorTrajectory()
        # Sub primitives
        list = trajectory.findSubPrimitives(beta=5)

        # Plot data
        if flag_plot:
            Parsing.__plot(sequence=sequence, label_list=list)
        if flag_save:
            Parsing.__save(label_list=list, path=path)

        return trajectory

    @staticmethod
    def __kalmanSmoother(original_sequence):
        """

        :param original_sequence:
        :return:
        """
        # Apply kalmar smooth to original sequence
        smoothed_sequence = Parsing.__initKalman(original_sequence[0]).smooth(original_sequence)[0]
        # Compute the mean square distance of original sequence with respect to the smoothed sequence
        distances = 0
        for index in range(0, len(original_sequence)):
            distances+= math.pow(MathUtils.distance(original_sequence[index], smoothed_sequence[index]),2)

        mean_square_distance = (math.sqrt(distances))/len(original_sequence)

        return smoothed_sequence, mean_square_distance

    @staticmethod
    def __initKalman(initial_point):
        """

        :param initial_point:
        :return:
        """
        # Kalman filther
        # specify parameters
        random_state = np.random.RandomState(0)

        # Transition matrix
        transition_matrix = [
            [1, 0],
            [0, 1]
        ]
        transition_offset = [0, 0]
        transition_covariance = np.eye(2)
        # Observation matrix
        observation_matrix = [
            [1, 0],
            [0, 1]
        ]
        observation_offset = [0, 0]
        observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
        # Initial state
        initial_state_mean = [initial_point[0], initial_point[1]]
        initial_state_covariance = [
            [1, 0],
            [0, 1]
        ]
        # Create Kalman Filter
        kalman_filter = KalmanFilter(
            transition_matrix, observation_matrix, transition_covariance,
            observation_covariance, transition_offset, observation_offset,
            initial_state_mean, initial_state_covariance,
            random_state=random_state
        )
        return kalman_filter

    @staticmethod
    def __plot(sequence, label_list):
        """

        :return:
        """
        # Plotting #
        fig, ax = plt.subplots(figsize=(10, 15))
        # plot original sequence
        original = plt.plot(sequence[:,0], sequence[:,1], color='b')
        # label
        for i in range(1, len(sequence)-1):
            ax.annotate(label_list[i-1], (sequence[i, 0], sequence[i, 1]))
        # legend
        #plt.legend((original[0]), ('sequence'), loc='lower right')
        plt.axis('equal')
        plt.show()

    @staticmethod
    def __save(label_list, path):
        """

        :param label_list:
        :param path:
        :return:
        """
        # Check parameters
        if not isinstance(path, str):
            raise TypeError
        if not isinstance(label_list, list):
            raise TypeError
        # Open and write file
        file = open(path, 'w')
        for item in label_list:
            file.write(item+"\n")










