from gesture import *
from model import *


gesture_models = [
    #Point(0,0) + Line(-2,-3) + Line(4,0)+ Line(-2,3), # triangle
    #Point(0,0) + Line(3,-3) + Line(0,3) + Line(-3,-3), # X
    #Point(0,0) + Line(0,-3) + Line(4,0) + Line(0, 3) + Line(-4,0), # rectangle
    #Point(0,0) + Arc(-3,-3, cw=False) + Arc(3,-3, cw=False) + Arc(3,3, cw=False) + Arc(-3,3, cw=False), # circle
    #Point(0,0) + Line(2, -2) + Line(4,6), # check
    #Point(0,0) + Line(2,3) + Line(2,-3), # caret
    #Point(0,0) + Arc(2,2) + Arc(2,-2) + Arc(-2,-2) + Line(0,-3), # question mark
    #Point(0,0) + Line(6,4) + Line(-3,0) + Line(4,1) + Line(-1, -3), # arrow
    #Point(0,0) + Line(-2,0) + Line(0,-4) + Line(2,0), # left square bracket
    #Point(0,0) + Line(2,0) + Line(0, -4)  + Line(-2, 0), # right square bracket
    #Point(0,0) + Line(2,-3) + Line(2,3), # V
    #Point(0,0) + Line(2, -3) + Line(-2,0) + Line(2,3), # delete
    #Point(0,0) + Arc(-2,-2, cw=False) + Line(0,-3) + Arc(-1,-1) + Arc(1,-1) + Line(0,-3) + Arc(2,-2,cw=False), # left curly brace
    #Point(0,0) + Arc(2,-2) + Line(0,-3) + Arc(1,-1, cw=False) + Arc(-1,-1, cw=False) + Line(0,-3) + Arc(-2,-2),  # right curly brace
    #Point(0,0) + Line(2,5) + Line(2, -5) + Line(-5, 3) + Line(6,0) + Line(-5, -3), # star
    #Point(0,0) + Arc(6,6, cw=False) + Arc(-1,1, cw=False) + Arc(-1,-1, cw=False) + Arc(6, -6, cw=False) # pigtail
]


baseDir  = '/Users/davide/Google Drive/Dottorato/Software/python/hmmtest/repository/'
trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'

mode = 2

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

if mode == 2:
    for gesture in gesture_models:
        factory = ClassifierFactory()
        factory.setLineSamplesPath(trainingDir)
        model, edges = factory.createClassifier(gesture)
        plot_gesture(model)

