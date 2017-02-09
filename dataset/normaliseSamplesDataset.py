import csv
import numpy
import scipy
import re
from math import sin, cos, radians
from .csvDataset import *

###
# This class defines the tools for making csv dataset (pre-processing, normalise and samples).
###

class ScaleDatasetTransform(DatasetTransform):
    def __init__(self, scale=100, cols=[0,1]):
        super()
        if not isinstance(cols, list):
            return TypeError
        self.cols = cols

        if  isinstance(scale, int):
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
        super()
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



class NormaliseLengthTransform(DatasetTransform):
    def __init__(self, axisMode=True, cols=[0,1]):
        super()
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

        for i in range(0, len(self.cols)):
            vmax = maxs[i]
            vmin = mins[i]

            if self.axisMode:
                den = vmax - vmin

            sequence[:, self.cols[i]] = sequence[:, self.cols[i]]  / den

        return sequence




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

    @staticmethod
    def create_folder(baseDir, gesture_name):
        # Folders
        if not os.path.exists(baseDir + 'original/' + gesture_name):
            os.makedirs(baseDir + 'original/' + gesture_name)
        if not os.path.exists(baseDir + 'normalised-trajectory/' + gesture_name):
            os.makedirs(baseDir + 'normalised-trajectory/' + gesture_name)
        if not os.path.exists(baseDir + 'down-trajectory/' + gesture_name):
            os.makedirs(baseDir + 'down-trajectory/' + gesture_name)