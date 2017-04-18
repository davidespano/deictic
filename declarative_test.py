from gesture import *
from model import *
from test import *

gesture_models = [
    (Point(0,0) + Line(-3,-4) + Line(6,0)+ Line(-3,4), 'triangle'), # triangle
    (Point(0,0) + Line(3,-3) + Line(0,3) + Line(-3,-3), 'x'), # X
    (Point(0,0) + Line(0,-3) + Line(4,0) + Line(0, 3) + Line(-4,0), 'rectangle'), # rectangle
    (Point(0,0) + Arc(-3,-3, cw=False) + Arc(3,-3, cw=False) + Arc(3,3, cw=False) + Arc(-3,3, cw=False), 'circle'), # circle
    (Point(0,0) + Line(2, -2) + Line(4,6), 'check'), # check
    (Point(0,0) + Line(2,3) + Line(2,-3), 'caret'), # caret
    (Point(0,0) + Arc(4,4) + Arc(4,-4) + Arc(-4,-4) + Arc(-2,-2, cw=False) + Arc(2, -2, cw=False), 'question_mark'),  # question mark
    (Point(0,0) + Line(6,4) + Line(-4,0) + Line(5,1) + Line(-1, -4), 'arrow'), # arrow
    (Point(0,0) + Line(-4,0) + Line(0,-5) + Line(4,0), 'left_sq_bracket'), # left square bracket
    (Point(0,0) + Line(4,0) + Line(0, -5)  + Line(-4, 0), 'right_sq_bracket'), # right square bracket
    (Point(0,0) + Line(2,-3) + Line(2,3), 'v'), # V
    (Point(0,0) + Line(2, -3) + Line(-2,0) + Line(2,3), 'delete_mark'), # delete_mark
    (Point(0,0) + Arc(-5,-5, cw=False) + Arc(-3,-3)  + Arc(3,-3) +  Arc(5,-5,cw=False),
     "left_curly_brace"),  # left curly brace
    (Point(0,0) + Arc(5,-5) +  Arc(3,-3, cw=False) + Arc(-3,-3, cw=False) + Arc(-5,-5), "right_curly_brace"),
    (Point(0,0) + Line(2,5) + Line(2, -5) + Line(-5, 3) + Line(6,0) + Line(-5, -3), 'star'), # star
    (Point(0,0) + Arc(3,3, cw=False) + Arc(-1,1, cw=False) + Arc(-1,-1, cw=False) + Arc(3, -3, cw=False), "pigtail") # pigtail
]

#(Point(0,0) + Arc(3,-3) + Arc(-3,-3) + Arc(-3,3) + Arc(3,3), 'circle'), # circle

#baseDir  = '/Users/davide/Google Drive/Dottorato/Software/python/hmmtest/repository/'
#baseDir = '/home/alessandro/PycharmProjects/deictic/repository/'
baseDir = '/Users/davide/PycharmProjects/deictic/repository/'

trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'
testDir = baseDir + "deictic/1dollar-dataset/resampled/"

mode = 4

if mode == 1:
    for gesture in gesture_models:
        processor = ModelPreprocessor(gesture)
        transform1 = CenteringTransform()
        transform2 = NormaliseLengthTransform(axisMode=True)
        transform3 = ScaleDatasetTransform(scale=100)
        processor.transforms.addTranform(transform1)
        processor.transforms.addTranform(transform2)
        processor.transforms.addTranform(transform3)
        print(gesture)
        processor.preprocess()
        gesture.plot()

if mode == 4:
    hmms = []
    names = []
    for gesture, name  in gesture_models:
        factory = ClassifierFactory()
        factory.states = 6
        factory.spu = 20
        factory.setLineSamplesPath(trainingDir)
        factory.setClockwiseArcSamplesPath(arcClockWiseDir)
        factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
        model, edges = factory.createClassifier(gesture)
        hmms.append(model)
        names.append(name)
    print(compares_deictic_models(hmms, testDir, names))


if mode == 3:
    dataset = CsvDataset(testDir + "right_sq_bracket/")
    #dataset.plot()
    #dataset.plot(singleMode=True)
    dataset.plot(sampleName='4_fast_right_sq_bracket_06.csv')

if mode == 5:
    #t = Point(0,0) + Line(4,0) + Point(2,0) + Line(0, -4)
    t = Point(0,0) + Arc(-6,-6, cw=False) + Line(0,-1) + Arc(-2,-2)  + Arc(2,-2) + Line(0,-1) + Arc(6,-6,cw=False)
    t.plot()


if mode == 6:
    for gesture, name in gesture_models:
        gesture.plot()

if mode == 7:
    #fork = Point(-2, 4) + Arc(2, -2, cw=False) + Arc(2, 2, cw=False) + Point(0, 4) + Line(0, -4)
    #fork.plot()

    dataset = CsvDataset(testDir + "pitchfork/")
    dataset.plot(count=5)

