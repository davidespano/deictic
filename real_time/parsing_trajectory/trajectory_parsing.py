from enum import Enum
# Imports
import numpy
# Kalman filter
from pykalman import KalmanFilter
# Math
import math
# Numpy
import numpy as np
# Plot
import matplotlib.pyplot as plt

class MathUtils():
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
        if not isinstance(sequence, numpy.ndarray):
            raise Exception("sequence must be a numpy array.")
        # Inizialization
        #
        self.__sequence = sequence
        #
        self.__labels = ["0" for x in range(len(sequence))]
        #
        self.__trajectory = numpy.zeros((len(sequence)),dtype=float)
        #
        self.__curvatures = [None for x in range(len(sequence))]

    # Methods #
    def algorithm1(self, threshold_a):
        """
            algorithm1 labels the sequence's points in accordance with the Algorithm 1 proposed in "Parsing 3D motion trajectory for gesture recognition".
        :param threshold_a:
        :return: list of label
        """
        threshold_a = 0.0028
        # Parsing line
        for t in range(1, len(self.__sequence) - 1, 1):
            # Compute delta
            num1 = MathUtils.sub(self.__sequence[t], self.__sequence[t - 1])
            num2 = MathUtils.sub(self.__sequence[t + 1], self.__sequence[t])
            num = MathUtils.dot(num1, num2)
            den1 = MathUtils.magn(MathUtils.sub(self.__sequence[t], self.__sequence[t - 1]))
            den2 = MathUtils.magn(MathUtils.sub(self.__sequence[t + 1], self.__sequence[t]))
            den = den1 * den2
            delta = 1 - (num / den)
            # Check delta
            if delta < threshold_a:
                self.__labels[t] = ("A")
        return self.__labels

    def algorithm2(self, threshold_b = None):
        """
            algorithm2 labels the sequence's points in accordance with the Algorithm 2 proposed in "Parsing 3D motion trajectory for gesture recognition".
        :param threshold_b:
        :return:
        """
        for t in range(1, len(self.__sequence)-2):
            if self.__labels[t] == "0" and self.__labels[t+1] == "0":
                # compute volume
                self.__labels[t] = "B"
                self.__labels[t + 1] = "B"
        return self.__labels

    def algorithm3(self):
        """
            algorithm3 provides to label the sequence's points, in accordance with the Algorithm 3 proposed in "Parsing 3D motion trajectory for gesture recognition"
            in order to localize the boundary points (isolated points or label transition).
        :param
        :return:
        """
        for t in range(2, len(self.__sequence)-2):
            if self.__labels[t] == "0":
                self.__labels[t] = "O" # isolated points
            elif self.__labels[t+1] != "0" and self.__labels[t] != self.__labels[t+1]:
                self.__labels[t] = "O" # for label transition
        return self.__labels

    def descriptorTrajectory(self):
        """

        :return:
        """
        #
        for t in range(2, len(self.__sequence)-2):
            if self.__labels[t] == "A":
                self.__trajectory[t] = 1
            elif self.__labels[t] == "B":
                _lambda = 1
                ### get descriptor ###
                # curvature
                k = self.__computeCurvature(t)
                # approssimation curvature
                k_s = self.__approximationCurvature(t)
                # descriptor
                self.__trajectory[t] = math.sqrt(math.pow(k,2) + _lambda*math.pow(k_s, 2))
        return self.__trajectory

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

    def quantization(self, beta):
        for i in self.__trajectory:
            print(i)


'''
    This class implements the class necessary to parse a trajectory into two primitives (straight line or plane arc) by labelling each frame. 
'''
class Parsing():
    # Attribute
    singleton = None
    __trajectories = []

    @staticmethod
    def getInstance():
        if Parsing.singleton == None:
            Parsing.singleton = Parsing()
        return Parsing.singleton


    # Methods #
    def parsingLine(self, sequence):
        """
            parsingLine provides to: apply a kalmar smoother to the sequence and label it in accordance with "Parsing 3D motion trajectory for gesture recognition"
        :param sequence:
        :return:
        """
        # Kalmar smoother and threshold for algorithm 1
        smoothed_sequence, threshold_a = self.__kalmanSmoother(sequence)
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
        # Quantization
        trajectory.quantization(beta=None)

        # append
        self.__trajectories.append(Trajectory)
        # Plot data
        self.__plot(original_sequence=sequence, smoothed_sequence=smoothed_sequence, label_list=list)
        return


    def __kalmanSmoother(self, original_sequence):
        """

        :param original_sequence:
        :return:
        """
        # Apply kalmar smooth to original sequence
        smoothed_sequence = self.__initKalman(original_sequence[0]).smooth(original_sequence)[0]
        # Compute the mean square distance of original sequence with respect to the smoothed sequence
        distances = 0
        for index in range(0, len(original_sequence)):
            distances+= math.pow(MathUtils.distance(original_sequence[index], smoothed_sequence[index]),2)

        mean_square_distance = (math.sqrt(distances))/len(original_sequence)

        return smoothed_sequence, mean_square_distance


    def __initKalman(self, initial_point):
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


    def __plot(self, original_sequence, smoothed_sequence, label_list):
        """

        :return:
        """
        # Plotting #
        fig, ax = plt.subplots(figsize=(10, 15))
        # plot original sequence
        original = plt.plot(original_sequence[:,0], original_sequence[:,1], color='b')
        # plot smoothed sequence
        smooth = plt.plot(smoothed_sequence[:, 0], smoothed_sequence[:,1], color='r')
        # label
        for i in range(1, len(smoothed_sequence)-1):
            ax.annotate(label_list[i-1], (smoothed_sequence[i, 0], smoothed_sequence[i, 1]))
        # legend
        plt.legend((original[0], smooth[0]), ('true', 'smooth'), loc='lower right')
        plt.axis('equal')
        plt.show()










