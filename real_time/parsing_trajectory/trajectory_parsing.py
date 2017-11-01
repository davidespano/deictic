# Imports
import numpy
# Kalman filter
from pykalman import KalmanFilter
# Math
import math

#
class Parsing():
    # Attribute
    singleton = None

    @staticmethod
    def getInstance():
        if Parsing.singleton == None:
            Parsing.singleton = Parsing()
        return Parsing.singleton

    def __init__(self):
        # specify parameters
        random_state = numpy.random.RandomState(0)
        transition_matrix = [[1, 0.01], [-0.1, 1]]
        transition_offset = [-0.1, 0.1]
        observation_matrix = numpy.eye(2) + random_state.randn(2, 2) * 0.01
        observation_offset = [1.0, -1.0]
        transition_covariance = numpy.eye(2)
        observation_covariance = numpy.eye(2) + random_state.randn(2, 2) * 0.01
        initial_state_mean = [80, -80]
        initial_state_covariance = [[1, 0.1], [-0.1, 1]]
        self.kalman_filter = KalmanFilter(
            transition_matrix, observation_matrix, transition_covariance,
            observation_covariance, transition_offset, observation_offset,
            initial_state_mean, initial_state_covariance,
            random_state=random_state
        )


    def parsingLine(self, sequence):
        """

        :param sequence:
        :return:
        """
        list = []

        # Kalmar smoother and threshold
        smoothed_sequence, threshold = self.__kalmanSmoother(sequence)
        #sequence = smoothed_sequence
        # Parsing line
        for t in range(1, len(sequence)-1, 1):

            # Compute delta
            num1 = self.__sub(sequence[t], sequence[t-1])
            num2 = self.__sub(sequence[t+1], sequence[t])
            num = self.__dot(num1,num2)
            den1 = self.__magn(self.__sub(sequence[t], sequence[t-1]))
            den2 = self.__magn(self.__sub(sequence[t+1], sequence[t]))
            den = den1 * den2

            delta = 1 - (num/den)
            # Check delta
            if delta < threshold:
                print("A")
                list.append("A")
            else:
                print("0")
                list.append("0")

            """
            # test collinear
            first_item = (sequence[t][1]-sequence[t-1][1])*(sequence[t+1][0]-sequence[t][0])
            second_item = (sequence[t+1][1]-sequence[t][1])*(sequence[t][0]-sequence[t-1][0])

            if first_item == second_item:
                print("A")
                list.append("A")
            else:
                print("0")
                list.append("0")
            """

        return list


    def __kalmanSmoother(self, original_sequence):
        """

        :param original_sequence:
        :return:
        """
        # Apply kalmar smooth to original sequence
        smoothed_sequence = self.kalman_filter.smooth(original_sequence)[0]
        # Compute the mean square distance of original sequence with respect to the smoothed sequence
        distances = 0
        for index in range(0, len(original_sequence)):
            distances+=self.__distance(original_sequence[index], smoothed_sequence[index])
            mean_square_distance = math.sqrt(distances/len(original_sequence))

        return smoothed_sequence, mean_square_distance


    def __magn(self, point):
        return math.sqrt(math.pow(point[0],2)+math.pow(point[1],2))
    def __dot(self, point_a, point_b):
        return point_a[0]*point_b[0] + point_a[1]*point_b[1]
    def __sub(self, point_a, point_b):
        vector = [point_a[0]-point_b[0], point_a[1]-point_b[1]]
        return vector
    def __distance(self, point_a, point_b):
        return math.hypot(point_b[0]-point_a[0], point_b[1]-point_a[1])
