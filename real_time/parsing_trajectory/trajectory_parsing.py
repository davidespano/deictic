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

class Point():
    def __init__(self, coordinates):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.label = None
        self.descriptor = None
    def setLabel(self, label):
        self.label = label
    def setDescriptor(self, descriptor):
        self.descriptor= descriptor

class Trajectory():
    def __init__(self, sequence):
        self.data = []
        for point in sequence:
            self.data.append(Point(point))



class Triangle():

    def __init__(self, point_a, point_b, point_c):
        self.point_a = point_a
        self.point_b = point_b
        self.point_c = point_c

    def curvature(self):
        self.side_a = Parsing.distance(point_a=self.point_a, point_b=self.point_b)
        self.side_b = Parsing.distance(point_a=self.point_b, point_b=self.point_c)
        self.side_c = Parsing.distance(point_a=self.point_c, point_b=self.point_a)
        surface = (self.side_a + self.side_b + self.side_c)/2
        self.curvature = (4 * math.sqrt(surface*(surface-self.side_a)*(surface-self.side_b)*(surface-self.side_c))) \
                   /(self.side_a*self.side_b*self.side_c)
        return self.curvature

class Tetrahedron():

    def __init__(self, point_a, point_b, point_c, point_d, point_e):
        self.prec_triangle = Triangle(point_a, point_b, point_c)
        self.actual_triangle = Triangle(point_b, point_c, point_d)
        self.next_triangle = Triangle(point_c, point_d, point_e)

    def curvature(self):
        return self.actual_triangle.curvature

    def approssimationCurvature(self):
        curvature_a = self.next_triangle
        curvature_b = self.prec_triangle
        self.approximations = 3*( (curvature_a - curvature_b)/(2*self.side_a+2*self.side_b+self.side_d+self.side_e))

'''
    This class implements the class necessary to parse a trajectory into two primitives (straight line or plane arc) by labelling each frame. 
'''
class Parsing():
    # Attribute
    singleton = None

    @staticmethod
    def getInstance():
        if Parsing.singleton == None:
            Parsing.singleton = Parsing()
        return Parsing.singleton



    def descriptionTrajectoryPrimitives(self, sequence):
        _lambda = 1# average number of zero points in the primitives as the lambda value of a given dataset
        for t in range(2, len(sequence)):
            if sequence[t] == "A":
                descriptor = (sequence, 1)
            elif sequence[t] == "B":
                tetrahedron = Tetrahedron(sequence[t-2], sequence[t-1], sequence[t], sequence[t+1], sequence[t+2])
                k = tetrahedron.curvature()
                ks = tetrahedron.approssimationCurvature()
                descriptor = math.sqrt(math.pow(k, 2) + _lambda*math.pow(ks,2))


    # Methods #
    def algorithm1(self, sequence, threshold_a):
        """
            algorithm1 labels the sequence's points in accordance with the Algorithm 1 proposed in "Parsing 3D motion trajectory for gesture recognition".
        :param sequence: original user's trajectory
        :return: list of label
        """
        list = []
        threshold_a = 0.002
        # Parsing line
        for t in range(1, len(sequence)-1, 1):

            # Compute delta
            num1 = Parsing.__sub(sequence[t], sequence[t-1])
            num2 = Parsing.__sub(sequence[t+1], sequence[t])
            num = Parsing.__dot(num1,num2)
            den1 = Parsing.__magn(self.__sub(sequence[t], sequence[t-1]))
            den2 = Parsing.__magn(self.__sub(sequence[t+1], sequence[t]))
            den = den1 * den2

            delta = 1-(num/den)
            # Check delta
            if delta < threshold_a:
                list.append("A")
            else:
                list.append("0")
        return list

    def algorithm2(self, sequence, list, threshold_b = None):
        """
            algorithm2 labels the sequence's points in accordance with the Algorithm 2 proposed in "Parsing 3D motion trajectory for gesture recognition".
        :param sequence: original user's trajectory
        :param list: list of label recived from algorithm1
        :return:
        """
        for t in range(1, len(list)-1):
            if list[t] == "0" and list[t+1] == "0":
                # compute volume
                list[t] = "B"
                list[t + 1] = "B"

        return list

    def algorithm3(self, list):
        """
            algorithm3 provides to label the sequence's points, in accordance with the Algorithm 3 proposed in "Parsing 3D motion trajectory for gesture recognition"
            in order to localize the boundary points (isolated points or label transition).
        :param list:
        :return:
        """
        for t in range(2, len(list)-1):
            if list[t] == "0":
                list[t] = "O" # isolated points
            elif list[t+1] != "0" and list[t] != list[t+1]:
                list[t] = "O" # for label transition

        return list


    def parsingLine(self, sequence):
        """
            parsingLine provides to: apply a kalmar smoother to the sequence and label it in accordance with "Parsing 3D motion trajectory for gesture recognition"
        :param sequence:
        :return:
        """
        # Kalmar smoother and threshold for algorithm 1
        smoothed_sequence, threshold_a = self.__kalmanSmoother(sequence)
        # Algorithm 1 (find straight linear)
        list = self.algorithm1(smoothed_sequence, threshold_a)
        # Algorithm 2 (find plane arc)
        list = self.algorithm2(smoothed_sequence, list)
        # Algorithm 3 (localizing boundary points)
        list = self.algorithm3(list)

        # Plot data
        self.__plot(original_sequence=sequence, smoothed_sequence=smoothed_sequence, label_list=list)


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
            distances+= math.pow(Parsing.__distance(original_sequence[index], smoothed_sequence[index]),2)

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









