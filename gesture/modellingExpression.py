from gesture.declarativeModel import ClassifierFactory
from config import Config
# Enum
from enum import Enum


class TypeRecognizer(Enum):
    online = 0
    offline = 1

class ModelFactory():

    @staticmethod
    def createHmm(expressions, num_states = 6, type=TypeRecognizer.offline, distance=6, num_samples = 20):
        """
            Given a deictic expression, this method returns its online or offline hmm model.
        :param expressions:
        :param num_states:
        :param type:
        :param distance:
        :param num_saples:
        :return:
        """
        # Check parameters
        if not isinstance(num_states, int):
            raise TypeError
        if not isinstance(type, TypeRecognizer):
            raise TypeError
        if not isinstance(distance, (int,float)):
            raise TypeError
        if not isinstance(num_samples, int):
            raise TypeError



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
            raise Exception("expressions must be a dictionary of sequences defined through deictic primitives:'gesture_name':[exp1, exp2, ...]")
        if not isinstance(num_states, int):
            raise Exception("num_states must be int.")
        if not isinstance(num_samples, int):
            raise Exception("num_samples must be int.")
        # Create models
        hmms = {}
        factory = ClassifierFactory()
        factory.setLineSamplesPath(Config.trainingDir)
        factory.setClockwiseArcSamplesPath(Config.arcClockWiseDir)
        factory.setCounterClockwiseArcSamplesPath(Config.arcCounterClockWiseDir)
        factory.states = num_states
        factory.spu = num_samples
        for gesture_label in expressions.keys():
            hmms[gesture_label] = []
            for expression in expressions[gesture_label]:
                model, edges = factory.createClassifier(expression)
                hmms[gesture_label].append(model)
        return hmms