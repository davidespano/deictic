'''
    Config class defines the set of main variable used in the project and relied on the position of the library.
'''

class Config():
    user = "ale"
    baseDir = '/home/'+user+'/PycharmProjects/deictic/repository/' # library path
    trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
    arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
    arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'
