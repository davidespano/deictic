from dataset import CsvDataset, RemovingFrames, ResampleInSpaceTransform
from real_time.scale_detection import ArtificialSequence
from config import Config
from real_time.parsing_trajectory.trajectory_parsing import MathUtils
from gesture.datasetExpressions import DatasetExpressions
from model import Point, Line, Arc
import os
import math
import numpy
import matplotlib.pyplot as plt

class Trajectory():
    def __init_(self, segments=[]):
        # check
        if not isinstance(segments, list) or not all(isinstance(item, Segment) for item in segments):
            raise TypeError
        # features
        self.segments = segments
        # sum of aggragated lengths
        self.__aggragated_lengths = sum([segment.getLength() for segment in self.segments])
        # mean of aggragated angles
        self.__aggragated_angles = sum([segment.getAngle() for segment in self.segments])/len(self.segments)
    def __init__(self, points=[], file_path=None):
        # check
        if not isinstance(points, numpy.ndarray):
            raise TypeError
        # create segments from frames
        self.__init_(Trajectory.__turning_angles(points))
        self.points = points
        self.file_path = file_path

    # public methods #
    def compare(self, other):
        # check
        if not isinstance(other, Trajectory):
            raise TypeError
        # compare
        not_matching_length = 0
        index_segment = 0
        if len(self.segments) <= len(other.segments):
            for seg in self.segments:
               if seg.getLength() > ((self.__aggragated_lengths*20)/100):
                   if not seg.equals(other=other.segments[index_segment], threeshold=(math.pi/4)):
                       not_matching_length += seg.getLength()
                       if not_matching_length > (self.__aggragated_lengths/10):
                           return False
                   else:
                       index_segment+=1
                       not_matching_length = 0
            return True
        return False

    def distance(self, other):
        # check
        if not isinstance(other, Trajectory):
            raise TypeError
        # comparise each segment with the input template
        comparisons = [self.segments[index].proportion(other.segments[index])
                       for index in range(len(self.segments))]
        # find scale
        scale = sum(comparisons)/len(comparisons)
        return scale

    def plot(self):
        # Plotting #
        # plot points
        plt.plot(self.points[:,0], self.points[:,1], color='b')
        # plot segments
        for segment in self.segments:
            point_label = segment.getEndPoint()
            label = str(math.degrees(segment.getAngle()))
            plt.annotate(label, (point_label[0],point_label[1]))
            plt.scatter(point_label[0], point_label[1])
        # legend
        plt.axis('equal')
        plt.show()

    @staticmethod
    def __turning_angles(array):
        segments = []
        # create and aggregate segments #
        for index in range(len(array)-1):
            # get points vector g
            point_a = array[index]
            point_b = array[index+1]

            # define segment
            new_segment = Segment(start=point_a, end=point_b)

            # aggregate the new segment if the difference is less than pi/8
            if len(segments)>0 and new_segment.equals(segments[-1], threeshold=math.pi/8):
                # aggregate the two last defined segments
                segments[-1] = Segment(segments[-1].start, point_b)
            else:
                # add new_segment
                segments.append(new_segment)


        return segments

class Segment():

    def __init__(self, start=0, end=0, referenced_seg = [1,0]):
        #
        if not all(isinstance(item, numpy.ndarray) for item in [start, end]):
            raise TypeError
        #
        self.start = start
        self.end = end
        self.vector = MathUtils.sub(self.start,self.end)
        self.vector_normalize = MathUtils.normalize(self.vector)
        self.referenced_seg = referenced_seg
        # parameters
        self.length = MathUtils.magn(self.vector)
        self.angle = self.__findAngle()

    # public methods #
    def equals(self, other, threeshold = (math.pi/8)):
        # check
        if not isinstance(other, Segment):
            raise TypeError
        max_value=2*math.pi
        values = [(self.getAngle()-other.getAngle()+max_value)%max_value,
                  (other.getAngle()-self.getAngle()+max_value)%max_value]
        difference = min(values)
        if difference < threeshold:
            return True
        return False
    def proportion(self, other):
        # check
        if not isinstance(other, Segment):
            raise TypeError
        # compute proportion
        return abs(round(self.__length / other.__length, 2))

    # private methods #
    def __findAngle(self):
        #angle = math.atan2(self.vector_normalize[1], self.vector_normalize[0]) - \
        #        math.atan2(self.referenced_seg[1], self.referenced_seg[0])

        seg = MathUtils.normalize(MathUtils.sub(self.start, self.end))
        vect = MathUtils.dot(self.referenced_seg, seg)
        angle = math.acos(vect)

        #if angle < 0:
        #    angle += 2 * math.pi
        return angle

    # get
    def getAngle(self):
        return self.angle
    def getLength(self):
        return self.length
    def getEndPoint(self):
        return self.end

base_dir = Config.baseDir+'deictic/1dollar-dataset/'
input_dir = base_dir+'raw/'
output_dir = base_dir+'template/'
filtered_dir = base_dir+'filtered/'

def test_sara():
    dictionary = CsvDataset(filtered_dir+'circle/')
    file = dictionary.readFile("1_medium_circle_01.csv")

    print("-------------POINTS-------------")
    for index_point in range(len(file)-1):
        print("--------------------------")
        a = file[index_point]
        b = file[index_point+1]
        segment = Segment(start=a, end=b)
        print("points: "+str(a) + ' - ' + str(b))
        print("Angle: "+ str(segment.getAngle()) + " in degrees: " + str(math.degrees(segment.getAngle())))

    print("\n\n\n\n")
    print("--------------SEGMENTS------------")
    trajectory = Trajectory(file)
    for segment in trajectory.segments:
        print("--------------------------")
        print("points: "+str(segment.start) + ' - ' + str(segment.end))
        print("Angle: "+ str(segment.getAngle()) + " in degrees: " + str(math.degrees(segment.getAngle())))
    trajectory.plot()


# Kalman database #
# for name in os.listdir(input_dir):
#     # Kalman filter #
#     dataset = CsvDataset(input_dir+name+'/')
#     transform_kalman = KalmanFilterTransform()
#     dataset.addTransform(transform_kalman)
#     dataset.applyTransforms(output_dir=output_dir+name+'/')
#     print(name + ' done!')
# templates naif #
#dictionary = DatasetExpressions.returnExpressions(DatasetExpressions.TypeDataset.unistroke_1dollar)
# dictionary = {  'circle_2': [(Point(0,0) +Arc(3,3,False)+Arc(-3,3,False))],
#                'circle_3': [(Point(0,0) +Arc(3,3,False)+Arc(-3,3,False)+Arc(-3,-3,False))],
#               'circle_4': [(Point(0,0) +Arc(3,3,False)+Arc(-3,3,False)+Arc(-3,-3,False))],
#                'rectangle':[(Point(-5,2.5)+Line(0,-5)+Line(10,0)+Line(0,5)+Line(-10,0))],
#             }
# elements=[]
# for key,value in dictionary.items():
#     artificial_trajectory = ArtificialSequence(expressions=value, spu=64)
#     artificial_trajectory.points = artificial_trajectory.points*100
#
#     if not output_dir is None and not os.path.exists(output_dir+key+'/'):
#         os.makedirs(output_dir+key+'/')
#
#     artificial_trajectory.save(file_path=output_dir+key+'/'+key+'.csv')
#     elements.append(artificial_trajectory)
#     #artificial_trajectory.plot()
#     sequence = Trajectory(artificial_trajectory.points)
#     for seg in sequence.segments:
#         print('Angle: '+str(seg.getAngle()) + ' - Start: ' + str(seg.start) + ' - End: '+str(seg.end))
#     print('\n')
#     sequence.plot()






    # sequence = Trajectory(file[0])
    # for seg in sequence.segments:
    #     print('Angle: '+str(seg.getAngle()) + ' - Start: ' + str(seg.start) + ' - End: '+str(seg.end))
    # print('\n')
    # sequence.plot()



#### turning angles ####
none_count = 0
for name in ['circle']:#os.listdir(output_dir):
    #### templates ####
    dictionary = CsvDataset(filtered_dir+'circle/')
    file = dictionary.readFile("1_medium_circle_01.csv")
    template = Trajectory(file)


    #### incompleted trajectories ####
    dataset_incompleted = CsvDataset(filtered_dir+name+'/')
    transform_removeFrames = RemovingFrames(stage=25)
    transform_resample = ResampleInSpaceTransform(samples=64)
    dataset_incompleted.addTransform(transform_removeFrames)
    dataset_incompleted.addTransform(transform_resample)
    incompleted_trajectories = []
    for file in dataset_incompleted.applyTransforms():
       incompleted_trajectories.append(Trajectory(points=file[0],file_path=file[1]))

    none_gesture = 0
    for incompleted in incompleted_trajectories:
        if incompleted.file_path != template.file_path and 'medium' in incompleted.file_path:
            if not incompleted.compare(template):
                none_gesture +=1
                #incompleted.plot()

    print(name + " None: "+str(none_gesture))
    none_count += none_gesture
print("Total None: "+str(none_count))