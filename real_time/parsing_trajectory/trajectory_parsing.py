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

    # Methods #
    def algorithm1(self, sequence):
        """
            algorithm1 labels the sequence's points in accordance with the Algorithm 1 proposed in "Parsing 3D motion trajectory for gesture recognition".
        :param sequence: original user's trajectory
        :return: list of label
        """
        list = []
        # Kalmar smoother and threshold
        smoothed_sequence, threshold = Parsing.__kalmanSmoother(sequence)
        threshold = 0.002
        print(threshold)
        # Parsing line
        for t in range(1, len(smoothed_sequence)-1, 1):

            # Compute delta
            num1 = Parsing.__sub(smoothed_sequence[t], smoothed_sequence[t-1])
            num2 = Parsing.__sub(smoothed_sequence[t+1], smoothed_sequence[t])
            num = Parsing.__dot(num1,num2)
            den1 = Parsing.__magn(self.__sub(smoothed_sequence[t], smoothed_sequence[t-1]))
            den2 = Parsing.__magn(self.__sub(smoothed_sequence[t+1], smoothed_sequence[t]))
            den = den1 * den2

            delta = 1-(num/den)
            # Check delta
            if delta < threshold:
                list.append("A")
            else:
                list.append("0")
        return smoothed_sequence,list

    def algorithm2(self, sequence, list):
        """
            algorithm1 labels the sequence's points in accordance with the Algorithm 2 proposed in "Parsing 3D motion trajectory for gesture recognition".
        :param sequence: original user's trajectory
        :param list: list of label recived from algorithm1
        :return:
        """




    def parsingLine(self, sequence):
        """

        :param sequence:
        :return:
        """
        # Algorithm 1 (find straight linear)
        smoothed_sequence, list = self.algorithm1(sequence)
        # Algorithm 2 (find plane arc)

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
    def __volTetrahedron(point_a, point_b, point_c):
        dimension = len(point_a)

        array = [[point_a[0],point_a[1],1],
                 [point_b[0],point_b[1],1],
                 [point_c[0],point_c[1],1]]
        det = array
        fact = math.pow(1/math.factorial(dimension), 3)

        volume = fact*det



    @staticmethod
    def __magn(self, point):
        return math.sqrt(math.pow(point[0],2)+math.pow(point[1],2))
    @staticmethod
    def __dot(self, point_a, point_b):
        return point_a[0]*point_b[0] + point_a[1]*point_b[1]
    @staticmethod
    def __sub(self, point_a, point_b):
        vector = [point_a[0]-point_b[0], point_a[1]-point_b[1]]
        return vector
    @staticmethod
    def __distance(self, point_a, point_b):
        return math.hypot(point_b[0]-point_a[0], point_b[1]-point_a[1])
