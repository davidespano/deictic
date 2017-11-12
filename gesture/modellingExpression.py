from model import *
from topology import *

# todo rivedere path primitive
#baseDir = '/home/sara/PycharmProjects/deictic/repository/'
baseDir = '/home/ale/PycharmProjects/deictic/repository/'
#baseDir = '/Users/davide/PycharmProjects/deictic/repository/'
trainingDir = baseDir + 'deictic/unica-dataset/raw/right/'
arcClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1ClockWise/'
arcCounterClockWiseDir = baseDir + 'deictic/unica-dataset/raw/arc1CounterClockWise/'

class ModelFactory():

    @staticmethod
    def createHmm(expressions, num_states = 6, num_samples = 20):
        """
            Given a deictic expression, this method returns its model.
        :param expression: the deictic expression which describes the model.
        :param num_states: the number of states of the new model.
        :param num_samples:
        :return: the model that implements the passed expression.
        """
        # Check parameters
        if not isinstance(expressions, dict):
            raise("expressions must be a dictionary of expresssions.")

        hmms = dict()
        factory = ClassifierFactory()
        factory.setLineSamplesPath(trainingDir)
        factory.setClockwiseArcSamplesPath(arcClockWiseDir)
        factory.setCounterClockwiseArcSamplesPath(arcCounterClockWiseDir)
        factory.states = n_states
        factory.spu = n_samples
        for gesture_label in expressions.keys():
            hmms[gesture_label] = []
            for expression in expressions[gesture_label]:
                model, edges = factory.createClassifier(expression)
                hmms[gesture_label].append(model)
            print('Group {0} created'.format(k))
        return hmms