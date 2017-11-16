from enum import Enum
# Kalman filter
from pykalman import KalmanFilter
# Math
import math
# Numpy
import numpy as np
# Plot
import matplotlib.pyplot as plt

class MathUtils():
    # todo: check parameters
    @staticmethod
    def findNearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx
    @staticmethod
    def surfaceTriangle(side_a, side_b, side_c):
        return (side_a+side_b+side_c)/2
    @staticmethod
    def curvature(side_a, side_b, side_c):
        surface = MathUtils.surfaceTriangle(side_a, side_b, side_c)
        return (4 * math.sqrt(surface * (surface - side_a) * (surface - side_b) * (surface - side_c))) \
                               / (side_a * side_b * side_c)
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





class Trajectory():

    class TypePrimitive(Enum):
        NONE = "0"
        LINE = "A"
        ARC = "B"
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
    def algorithm1(self, threshold_a):
        """
            algorithm1 labels the sequence's points in accordance with the Algorithm 1 proposed in "Parsing 3D motion trajectory for gesture recognition".
        :param threshold_a:
        :return: list of label
        """
        #threshold_a = 0.009
        # Parsing line
        for t in range(1, len(self.__sequence) - 1, 1):
            # Compute delta
            # num1 = MathUtils.sub(self.__sequence[t], self.__sequence[t - 1])
            # num2 = MathUtils.sub(self.__sequence[t + 1], self.__sequence[t])
            # num = MathUtils.dot(num1, num2)
            # den1 = MathUtils.magn(MathUtils.sub(self.__sequence[t], self.__sequence[t - 1]))
            # den2 = MathUtils.magn(MathUtils.sub(self.__sequence[t + 1], self.__sequence[t]))
            # den = den1 * den2
            # delta = 1 - (num / den)

            # compute area
            a = self.__sequence[t-1]
            b = self.__sequence[t]
            c = self.__sequence[t + 1]
            point_x = [a[0], b[0], c[0]]
            point_y = [a[1], b[1], c[1]]
            max_x = max(point_x)
            min_x=min(point_x)
            max_y=max(point_y)
            min_y=min(point_y)
            area = (a[0] * (b[1]-c[1])) + (b[0] * (c[1]-a[1])) + (c[0] * (a[1]-b[1]))
            delta = math.fabs(area/2) / math.fabs((max_x-min_x) * (max_y-min_y))

            #print(str(max_x) + " - " + str(min_x)+ " - " + str(max_y)+ " - " + str(min_y)+" - " + str(delta))
            # Check delta
            if delta < threshold_a:
                self.__labels[t] = (Trajectory.TypePrimitive.LINE.value)
        return self.__labels

    def algorithm2(self, threshold_b = None):
        """
            algorithm2 labels the sequence's points in accordance with the Algorithm 2 proposed in "Parsing 3D motion trajectory for gesture recognition".
        :param threshold_b:
        :return:
        """
        for t in range(1, len(self.__sequence)-2):
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
        for t in range(2, len(self.__sequence)-2):
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
        for t in range(2, len(self.__sequence)-2):
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
        # Group labels
        # self.__groupLabels()
        # # Subprimitives
        # for primitive in self.__primitives:
        #     primitive.quantizationIntervals(beta=beta)
        # return self.__primitives
        indexes = []
        indexes.append(1)
        for index in range(2, len(self.__sequence) - 1):
            if self.__labels[index] != self.__labels[index - 1] or index == len(self.__sequence)-2:
                self.__quantizationIntervals(indexes=indexes, beta=beta)
                indexes.clear()
            indexes.append(index)
        return self.__labels

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

    def __groupLabels(self, beta):
        """

        :return:
        """
        self.__primitives = []
        points = []
        points.append(self.__sequence[1])
        for index in range(2, len(self.__sequence) - 1):
            if self.__labels[index] != self.__labels[index - 1]:
                #self.__primitives.append(Primitive(points=points, label=self.__labels[index-1]))

                points.clear()
            points.append(index)

    def __quantizationIntervals(self, indexes=[], beta=1):
        """

        :param beta:
        :return:
        """
        points = []
        for index in indexes:
            points.append(self.__descriptors[index])
        # Get min and max value from its points
        min_value = min(points)
        max_value = max(points)
        # quantization
        interval_value = (max_value-min_value)/beta
        interval_values = [min_value+(interval_value*x) for x in range(beta)]
        # assign interval
        for index in indexes:
            interval_index = MathUtils.findNearest(interval_values, self.__descriptors[index])
            self.__labels[index] = self.__labels[index]+str(interval_index)


'''
    This class implements the class necessary to parse a trajectory into two primitives (straight line or plane arc) by labelling each frame. 
'''
class Parsing():

    # Methods #
    @staticmethod
    def parsingLine(sequence):
        """
            parsingLine provides to: apply a kalmar smoother to the sequence and label it in accordance with "Parsing 3D motion trajectory for gesture recognition"
        :param sequence:
        :return:
        """
        # Kalmar smoother and threshold for algorithm 1
        smoothed_sequence, threshold_a = Parsing.__kalmanSmoother(sequence)
        # trajectory
        trajectory = Trajectory(smoothed_sequence)
        # Algorithm 1 (find straight linear)
        list = trajectory.algorithm1(threshold_a=threshold_a)
        # Algorithm 2 (find plane arc)
        list = trajectory.algorithm2(threshold_b=None)
        # Algorithm 3 (localizing boundary points)
        list = trajectory.algorithm3()
        # Descriptor
        f = trajectory.descriptorTrajectory()
        # Sub primitives
        #list = trajectory.findSubPrimitives(beta=4)

        # Plot data
        #Parsing.__plot(original_sequence=sequence, smoothed_sequence=smoothed_sequence, label_list=list)
        return list

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
    def __plot(original_sequence, smoothed_sequence, label_list):
        """

        :return:
        """
        # Plotting #
        fig, ax = plt.subplots(figsize=(10, 15))
        # plot original sequence
        #original = plt.plot(original_sequence[:,0], original_sequence[:,1], color='b')
        # plot smoothed sequence
        smooth = plt.plot(smoothed_sequence[:, 0], smoothed_sequence[:,1], color='r')
        # label
        for i in range(1, len(smoothed_sequence)-1):
            ax.annotate(label_list[i-1], (smoothed_sequence[i, 0], smoothed_sequence[i, 1]))
        # legend
        #plt.legend((original[0], smooth[0]), ('true', 'smooth'), loc='lower right')
        plt.axis('equal')
        plt.show()










