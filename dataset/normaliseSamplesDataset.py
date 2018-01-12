import csv
import numpy as np
import scipy
import scipy.signal
import re
import math
from copy import copy, deepcopy
from math import sin, cos, radians
from .csvDataset import *
from .geometry import *
# Kalman filter
from pykalman import KalmanFilter
# Parsing
from real_time.parsing_trajectory.trajectory_parsing import Parsing

###
# This class defines the tools for making csv dataset (pre-processing, normalise and samples).
###

class ScaleDatasetTransform(DatasetTransform):
    def __init__(self, scale=100, cols=[0,1]):
        super(ScaleDatasetTransform, self).__init__()
        if not isinstance(cols, list):
            return TypeError
        self.cols = cols

        if  isinstance(scale, float) or isinstance(scale, int):
            self.scale = []
            for i in range(0, len(cols)):
                self.scale.append(scale)
        if isinstance(scale, list):
            self.scale = scale


    def transform(self, sequence):
        if not isinstance(sequence, numpy.ndarray):
            raise TypeError

        for i in range(0, len(self.cols)):
            seq_index = self.cols[i]
            sequence[:, seq_index] = sequence[:, seq_index] * self.scale[i]

        return sequence


class CenteringTransform(DatasetTransform):
    def __init__(self, cols=[0, 1]):
        super(CenteringTransform, self).__init__()
        if not isinstance(cols, list) or len(cols) > 3 or len(cols) < 2:
            return TypeError
        self.cols = cols

    def transform(self, sequence):
        maxs = numpy.amax(sequence[:, self.cols], axis=0)
        mins = numpy.amin(sequence[:, self.cols], axis=0)

        for i in range(0, len(self.cols)):
            vmax = maxs[i]
            vmin = mins[i]

            sequence[:, self.cols[i]] = sequence[:, self.cols[i]] - 0.5 * (vmin + vmax)

        return sequence

class TranslateTransform(DatasetTransform):

    def __init__(self, cols=[0,1], t=[0,0]):
        super(TranslateTransform, self).__init__()
        if not isinstance(cols, list):
            return TypeError
        self.cols = cols

        if not isinstance(t, list) or len(t) < 2 :
            return TypeError
        self.t = t

    def transform(self, sequence):
        if not isinstance(sequence, numpy.ndarray):
            raise TypeError
        for i in range(0, len(self.cols)):
            sequence[:, self.cols[i]] = sequence[:, self.cols[i]] + self.t[i]
        return sequence

class NormaliseLengthTransform(DatasetTransform):
    def __init__(self, axisMode=True, cols=[0,1]):
        super(NormaliseLengthTransform, self).__init__()
        if not isinstance(axisMode, bool):
            return TypeError
        self.axisMode = axisMode

        if not isinstance(cols, list):
            return TypeError
        self.cols = cols

    def transform(self, sequence):

        maxs = numpy.amax(sequence[:, self.cols], axis=0)
        mins = numpy.amin(sequence[:, self.cols], axis=0)

        den = max(maxs-mins)

        if 0 in maxs-mins:
            self.axisMode = False

        for i in range(0, len(self.cols)):
            vmax = maxs[i]
            vmin = mins[i]

            if self.axisMode:
                den = vmax - vmin

            sequence[:, self.cols[i]] = sequence[:, self.cols[i]] / den

        return sequence


class SwapTransform(DatasetTransform):
    def __init__(self, cols=[[0,1],[1,0]]):
        super(SwapTransform, self).__init__()
        if not isinstance(cols, list):
            return TypeError
        self.cols = cols

    def transform(self, sequence):
        if not isinstance(sequence, numpy.ndarray):
            raise TypeError

        sequence_temp = deepcopy(sequence)
        for i in range(0, len(self.cols)):
            seq_index = self.cols[i][0]
            seq_swap = self.cols[i][1]
            sequence[:, seq_index] = sequence_temp[:,seq_swap]

        return  sequence

class RotateTransform(DatasetTransform):

    degrees = 1
    radians = 2

    def __init__(self, traslationMode = False, cols=[0,1], theta=0, centre = [0,0], unit= degrees):
        super(RotateTransform, self).__init__()
        if not isinstance(traslationMode, bool):
            return TypeError
        self.traslationMode = traslationMode
        if not isinstance(cols, list):
            return TypeError
        self.cols = cols
        if not isinstance(theta, float) and not isinstance(theta, int):
            return TypeError
        self.theta = theta
        if not isinstance(centre, list) or len(centre) != 2 :
            return TypeError
        self.centre = centre
        if unit != RotateTransform.degrees and unit != RotateTransform.radians:
            return TypeError
        self.unit = unit

    def transform(self, sequence):
        if not isinstance(sequence, numpy.ndarray):
            raise TypeError

        # Initialization matrix rotate
        if self.unit == RotateTransform.degrees:
            theta = numpy.radians(self.theta)
        else:
            theta = self.theta
        c, s = numpy.cos(theta), numpy.sin(theta)
        R = numpy.matrix('{} {} {}; {} {} {}; {} {} {}'.format(c,-s,0,s,c,0,0,0,1))
        matrix = deepcopy(R)

        # Initialization matrix translation and traslation back
        n = len(self.cols)+1
        matrix_translantion = numpy.zeros((n,n))
        matrix_translantion_back = numpy.zeros((n,n))
        i,j = numpy.indices(matrix_translantion.shape)

        if self.traslationMode == True: # shortcut for rotating around the centre of the box.
            maxs = numpy.amax(sequence[:, self.cols], axis=0)
            mins = numpy.amin(sequence[:, self.cols], axis=0)
            den = max(maxs-mins)

            self.centre[0] = den
            self.centre[1] = den

        if self.centre[0] != 0 and self.centre[1] != 0: # the rotation centre is not the origin
            matrix_translantion[0, 2] = self.centre[0]
            matrix_translantion_back[0, 2] = - self.centre[0]
            matrix_translantion[1, 2] = self.centre[1]
            matrix_translantion_back[1, 2] = - self.centre[1]
            matrix_translantion[i == j] = 1
            matrix_translantion_back[i == j] = 1

            # we compute the rotation around a generic center with a translation, rotation and a translation back
            matrix = matrix_translantion * R * matrix_translantion_back

        tmp = numpy.zeros((len(sequence), 3))
        tmp[:, 0] = sequence[:, self.cols[0]]
        tmp[:, 1] = sequence[:, self.cols[1]]
        tmp[:, 2] = 1

        tmp =  matrix * numpy.matrix(tmp).T

        tmp = tmp.T[:,:2]
        sequence[:,self.cols[0]] = numpy.squeeze(numpy.asarray(tmp[:,0]))
        sequence[:,self.cols[1]] = numpy.squeeze(numpy.asarray(tmp[:,1]))

        return sequence


class RotateCenterTransform(DatasetTransform):
    degrees = 1
    radians = 2
    def __init__(self, traslationMode = False, cols=[0,1], theta=0, centre = [0,0], unit= degrees):
        super(RotateCenterTransform, self).__init__()
        if not isinstance(traslationMode, bool):
            return TypeError
        self.traslationMode = traslationMode
        if not isinstance(cols, list):
            return TypeError
        self.cols = cols
        if not isinstance(theta, float) and not isinstance(theta, int):
            return TypeError
        self.theta = theta
        if not isinstance(centre, list) or len(centre) != 2 :
            return TypeError
        self.centre = centre

        if unit != RotateTransform.degrees and unit != RotateTransform.radians:
            return TypeError
        self.unit = unit

    def transform(self, sequence):
        if not isinstance(sequence, numpy.ndarray):
            raise TypeError

            # First point
        px = sequence[0, self.cols[0]]
        py = sequence[0, self.cols[1]]
        # Centroid
        centroid_x, centroid_y = Geometry2D.Centroid(sequence)
        # Degree
        deltaX = px - centroid_x
        deltaY = py - centroid_y

        # if(deltaX == 0):
        #     if(deltaY > 0):
        #         angleInDegrees =  90
        #     else:
        #         angleInDegrees =  -90
        # else:
        #     angleInDegrees = numpy.arctan(deltaY/deltaX) * 180 / numpy.pi

        d = abs(Geometry2D.distance(centroid_x, centroid_y, px, py))
        v = [centroid_x + d, centroid_y]
        u = [px - centroid_x, py - centroid_y]
        d1 = abs(Geometry2D.distance(0, 0, v[0], v[1]))
        d2 = abs(Geometry2D.distance(0, 0, u[0], u[1]))



        angleInDegrees = numpy.arccos(numpy.dot(u, v)/ (d1*d2))


        self.theta = -angleInDegrees
        self.unit = RotateTransform.radians
        self.centre = [centroid_x, centroid_y]
        self.traslationMode = False
        return RotateTransform.transform(self, sequence)


class Sampling(DatasetTransform):
    def __init__(self, scale=100):
        super(Sampling, self).__init__()
        if not isinstance(scale, int):
            return TypeError
        self.scale = scale

    def transform(self, sequence):
        if not isinstance(sequence, numpy.ndarray):
            raise TypeError

        element = (round(len(sequence)/self.scale))
        #sequence = sequence[0:len(sequence):element]
        a = scipy.signal.resample(sequence, self.scale+2)
        sequence = a[1:,]

        return sequence

class ResampleInSpaceTransformMultiStroke(DatasetTransform):
    def __init__(self, samples=20, cols=[0,1], strokes=None, index_stroke = -1):
        if not isinstance(samples, int):
            return TypeError
        self.samples = samples
        if not isinstance(cols, list):
            return TypeError
        self.cols = cols
        if not isinstance(strokes, int):
            return TypeError
        self.strokes = strokes
        if not isinstance(index_stroke, int):
            return TypeError
        self.index_stroke = index_stroke

    def transform(self, sequence):
        if not isinstance(sequence, numpy.ndarray):
            raise TypeError
        srcPts = numpy.copy(sequence).tolist()
        resampled = []

        for index in range(0, self.strokes):
            sequence = [x for x in srcPts if x[-1] == index+1]
            self.stroke = index + 1
            # Resampled
            if(len(sequence) > 1):
                sequence = ResampleInSpaceTransform.transform(self, numpy.array(sequence))
            for seq in sequence:
                resampled.append(seq)

        return resampled

class ResampleInSpaceTransform(DatasetTransform):
    def __init__(self, samples=20, cols=[0,1]):
        if not isinstance(samples, int):
            return TypeError
        self.samples = samples
        if not isinstance(cols, list):
            return TypeError
        self.cols = cols
        self.stroke = None

    def transform(self, sequence):
        # adapted from WobbrockLib
        if not isinstance(sequence, numpy.ndarray):
            raise TypeError
        srcPts = numpy.copy(sequence).tolist()
        length = Geometry2D.pathLength(sequence)
        size = len(srcPts)
        step = length/ max(self.samples -1, 1)

        resampled = []

        if self.stroke is None:
            resampled.append([srcPts[0][self.cols[0]], srcPts[0][self.cols[1]]])
        else:
            resampled.append([srcPts[0][self.cols[0]], srcPts[0][self.cols[1]], self.stroke])

        D = 0.0
        j = 1
        i = 1
        while i < len(srcPts):
            pt1x = srcPts[i - 1][self.cols[0]]
            pt1y = srcPts[i - 1][self.cols[1]]
            pt2x = srcPts[i][self.cols[0]]
            pt2y = srcPts[i][self.cols[1]]

            d = Geometry2D.distance(pt1x, pt1y, pt2x, pt2y) # distance in space

            if (D + d) >= step and d > 0: # has enough space been traversed in the last step?
                qx = pt1x + ((step - D) / d) * (pt2x - pt1x) # interpolate position
                qy = pt1y + ((step - D) / d) * (pt2y - pt1y) # interpolate position

                if self.stroke is None:
                    resampled.append([qx, qy])
                else:
                    resampled.append([qx, qy, self.stroke])

                srcPts.insert(i, [qx, qy]) # insert 'q' at position i in points s.t. 'q' will be the next i


                D = 0.0
            else:
                D+= d
            i +=1

        if D > 0.0:
            size = len(srcPts)
            if self.stroke is None:
                resampled.append([srcPts[size -1][0], srcPts[size -1][1]])
            else:
                resampled.append([srcPts[size - 1][0], srcPts[size - 1][1], self.stroke])

        return numpy.array(resampled)

# Resampling other version
class ResampleTransform(DatasetTransform):
    """
        this funciton resamples the sequences. Starting from computing the distance between the nth point and the previous (n-1), ResampleTransform checks if that distance is higher than delta or less:
        in the first case the nth point is kept and still remains in the sequence, in the other case the nth point is deleted.
    """
    def __init__(self, delta=0, cols=[0,1]):
        # Check parameters
        if isinstance(delta, (bool, str, list, dict)):
            raise TypeError
        if not isinstance(cols, list):
            raise TypeError
        self.__cols = cols
        self.__delta = delta

    # Public methods
    def transform(self, sequence):
        # Check parameter
        if not isinstance(sequence, numpy.ndarray):
            raise TypeError

        # initiliaze
        new_sequence = [sequence[0]]#sequence.tolist()
        last_index = 0
        #c = len(sequence)
        for index in range(1, len(sequence)):
            # take points
            point_a = new_sequence[last_index]
            point_b = sequence[index][self.__cols]
            distance = math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
            if distance > self.__delta:
                new_sequence.append(sequence[index])
                last_index+=1
        # for item in sequence:
        #     # take points
        #     point_a = self.__getPoint(sequence[sequence.index(item)-1])
        #     point_b = self.__getPoint(item)
        #     # distance
        #     distance = math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
        #     if distance < self.__delta:
        #         sequence.remove(item)
        #d = len(sequence)
        #print("Before: "+str(c)+" Ater: "+str(d))
        return numpy.array(new_sequence)

    # Private methods
    def __getPoint(self, frame):
        point = []
        for index in self.__cols:
            point.append(frame[index])
        return point

# Kalman
class KalmanFilterTransform(DatasetTransform):
    def __init__(self):
        self.dir = None

    def transform(self, sequence):
        # Check parameters
        # Apply kalmar smooth to original sequence
        original_sequence = sequence[:,[0,1]]
        return numpy.array(KalmanFilterTransform.setFilter(original_sequence[0]).smooth(original_sequence)[0])

    @staticmethod
    def setFilter(initial_point):
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
        kalman = KalmanFilter(
            transition_matrix, observation_matrix, transition_covariance,
            observation_covariance, transition_offset, observation_offset,
            initial_state_mean, initial_state_covariance,
            random_state=random_state
        )
        return kalman

# Parsing trajectory
class ParseSamples(DatasetTransform):
    #
    def __init__(self):
        self.dir=None
    #
    def transform(self, sequence):
        #return Parsing.parsingLine(sequence=sequence[:,[0,1]]).getLabelsSequence()
        return Parsing.parsingLine(sequence=sequence[:, [0, 1]]).getLabels()

class NormaliseSamples:

    #
    def __init__(self, dir):
        self.dir = dir

    # Get all csv files from directory
    def getCsvDataset(self):
        return DatasetIterator(self.dir)

######### Pre-processing original files dataset #########

    ## Swap
    # Is used to make up and down csv files from right and left csv files.
    def swap(self, output_dir, name, dimensions = 2):

        for filename in self.getCsvDataset():
            items = re.findall('\d*\D+', filename)# filename

            # Read file
            with open(self.dir + filename, "r") as f:
                reader = csv.reader(f, delimiter=',')
                vals = list(reader)
                result = numpy.array(vals).astype('float') # This array contains all file's data

                # X and Y
                if dimensions == 2:
                    # Swap x with y
                    for index in range(0, len(result)):
                        temp = result[index][0]
                        result[index][0] = result[index][1]
                        result[index][1] = temp
                # X, Y and Z
                elif dimensions == 3:
                    # Swap x with z
                    for index in range(0, len(result)):
                        temp = result[index][0]
                        result[index][0] = result[index][2]
                        result[index][2] = temp

                # Save file
                numpy.savetxt(output_dir + name + '_' + items[len(items)-1], result, delimiter=',')

    ## Rotate Lines
    # Is used to make diagonal csv files from rotating right and left files.
    def rotate_lines(self, output_dir, name, degree = 0):

        for filename in self.getCsvDataset():
            items = re.findall('\d*\D+', filename)# filename

            # Read file
            with open(self.dir + filename, "r") as f:
                reader = csv.reader(f, delimiter=',')
                vals = list(reader)
                result = numpy.array(vals).astype('float')

                # Compute max and min for each column
                maxs = result.argmax(axis=0);
                mins = result.argmin(axis=0);
                x_max = result[maxs[0], 0]
                y_max = result[maxs[1], 1]
                z_max = result[maxs[2], 2]
                x_min = result[mins[0], 0]
                y_min = result[mins[1], 1]
                z_min = result[mins[2], 2]
                den_x = (x_max + x_min)/2
                den_y = (y_max + y_min)/2
                den_z = (z_max + z_min)/2

                # Make translation and rotation
                theta = radians(degree)
                cosang, sinang = cos(theta), sin(theta)
                matrix_translantion = numpy.asmatrix(numpy.array(
                    [[1, 0, den_x],
                     [0, 1, den_y],
                     [0, 0, 1]]))
                matrix_translantion_back = numpy.asmatrix(numpy.array(
                    [[1, 0, -den_x],
                     [0, 1, -den_y],
                     [0, 0, 1]]))
                matrix_rotate = numpy.asmatrix(numpy.array(
                    [[cosang, - sinang, 0],
                     [sinang, cosang, 0],
                     [0, 0, 1]]))

                m = matrix_translantion * matrix_rotate * matrix_translantion_back;

                # For each frame add new value
                for index in range(0, len(result)):
                    result_temp = numpy.array([[0,0,1]])
                    result_temp[0][0]= result[index][0]
                    result_temp[0][1]= result[index][1]
                    t =  m * numpy.matrix(result_temp[0]).T
                    result[index][0] = t[0]
                    result[index][1] = t[1]

                # Save file
                numpy.savetxt(output_dir + name + '_' + items[len(items)-1], result, delimiter=',')


######### Normalise #########

    ## Normalise
    # Is used for normalising the original csv file
    def normalise(self, output_dir, norm_axis = False):

        for filename in self.getCsvDataset():

            # Read file
            with open(self.dir + filename, "r") as f:
                reader = csv.reader(f, delimiter=',')
                vals = list(reader)
                result = numpy.array(vals).astype('float')

                # Get max and min for each column
                maxs = result.argmax(axis=0);
                mins = result.argmin(axis=0);
                # Max
                x_max = result[maxs[0], 0]
                y_max = result[maxs[1], 1]
                z_max = result[maxs[2], 2]
                # Min
                x_min = result[mins[0], 0]
                y_min = result[mins[1], 1]
                z_min = result[mins[2], 2]

                # X Y Z
                if norm_axis:
                    den_x = x_max - x_min
                    den_y = y_max - y_min
                    den_z = z_max - z_min

                    result[:,0] = (result[:, 0] - x_min) / den_x
                    result[:,1] = (result[:, 1] - y_min) / den_y
                    result[:,2] = (result[:, 2] - z_min) / den_z

                    #numpy.savetxt(output_dir + filename, result, delimiter=',')

                else:
                    den = max(x_max - x_min, y_max - y_min, z_max - z_min);

                    # Normalise X, Y and Z
                    result[:, 0] = (result[:, 0] - x_min) / den
                    result[:, 1] = (result[:, 1] - y_min) / den
                    result[:, 2] = (result[:, 2] - z_min) / den

                    # Computes Delta X, Y and Z
                    #for i in range(1, result[:, 0].size):
                    #    result[i, 0] = result[i, 0] - result[i-1, 0]
                    #    result[i, 1] = result[i, 1] - result[i-1, 1]


                numpy.savetxt(output_dir + filename, result, delimiter=',')


######### Samples #########
    ## down_sample
    # Is used to sampling normalised data.
    def down_sample(self, output_dir, samples):

        # For each csv file
        for filename in self.getCsvDataset():

            # Read file
            with open(self.dir + filename, "r") as f:
                reader = csv.reader(f, delimiter=',')
                vals = list(reader)
                result = numpy.array(vals).astype('float')
                R = int(1.0 * result[:, 0].size / samples)
                a = numpy.zeros((samples - 1, int(result[0, :].size)))

                # Sampling
                for i in range(0, samples - 1):
                    start = i * R
                    end = ((i + 1) * R)
                    a[i, 0] = scipy.nanmean(result[start:end, 0])
                    a[i, 1] = scipy.nanmean(result[start:end, 1])
                    a[i, 2] = scipy.nanmean(result[start:end, 2])

                # Save file
                numpy.savetxt(output_dir + filename, a, delimiter=',')