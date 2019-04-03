# classifier factory
from gesture.declarativeModel import ClassifierFactory
from gesture.declarativeModelExt import ClassifierFactoryExt
# gesture type
from model.gestureModel import GestureExp

class CreateRecognizer():

    @staticmethod
    def createHMMs(expressions, num_states = 6, spu = 20):
        """
            Given a deictic expression, this method returns its parsed or offline hmm model.
        :param expressions: the deictic expressions which describes the models.
        :param num_states: the number of states of the new model.
        :param spu: samples per unit
        :return: models that implements the passed expression.
        """
        # check
        if not isinstance(expressions, dict):
            raise TypeError('expression must be a dictionary of list expressions')
        if not isinstance(num_states, int):
            raise TypeError
        if not isinstance(spu, int):
            raise TypeError
        # Classifier factory
        factory = ClassifierFactory(num_states=num_states, spu=spu)
        # create hmms for each label contained into expressions
        models = {}
        for gesture_label, values in expressions.items():
            models[gesture_label] = [CreateRecognizer.createHMM(expression=expression, factory=factory)
                                     for expression in values]
        return models


    @staticmethod
    def createHMM(expression, factory=ClassifierFactory(num_states=6, spu=20)):
        """
            starting from a deictic expression and a classifier factory, this function returns the hmm which recognizes
            the given expression.
        :param expression:
        :param factory:
        :return:
        """
        # check
        if not isinstance(expression, GestureExp):
            raise TypeError("expression must be a GestureExp")
        if not isinstance(factory, ClassifierFactory):
            raise TypeError("factory must be a ClassifierFactory")
        # create factory and hmm
        model, states = factory.createClassifier(expression)
        return model

