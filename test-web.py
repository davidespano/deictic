from gesture import *
import numpy as numpy

parser = StringParser()

baseDir = '/Users/davide/PycharmProjects/deictic/repository/'
trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'

triangleExpr = parser.fromString('P(0,0) + L(-3,-4) + L(6,0)+ L(-3,4)')
rectangleExpr = parser.fromString('P(0,0) + L(0,-3) + L(4,0) + L(0, 3) + L(-4,0)')


factory = ClassifierFactory(type=TypeRecognizer.offline)
factory.setLineSamplesPath(trainingDir)
factory.setClockwiseArcSamplesPath(arcClockWiseDir)
factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
factory.states = 16
factory.spu = 20

triangle = factory.createClassifier(triangleExpr);
rectangle = factory.createClassifier(rectangleExpr);

sample = triangle[0].sample();

print(sample)
print(numpy.exp(triangle[0].log_probability(sample)/len(sample)))
print(numpy.exp(rectangle[0].log_probability(sample)/len(sample)))


