# classifier factory
from gesture.declarativeModel import ClassifierFactory
# recognizer type
from model.gestureModel import TypeRecognizer, GestureExp, CompositeExp

class ModelExpression():

    @staticmethod
    def generatedModels(expressions, type=TypeRecognizer.offline, num_states = 6, spu = 20):
        """
            Given a deictic expression, this method returns its online or offline hmm model.
        :param expressions: the deictic expressions which describes the models.
        :param type:
        :param num_states: the number of states of the new model.
        :param spu: samples per unit
        :return: the models that implements the passed expression.
        """
        # Check parameters
        if not isinstance(expressions, dict) or not all( (all(isinstance(exp,(GestureExp, CompositeExp)) for exp in expression) for expression in expressions.values())):
            raise TypeError
        if not isinstance(type, TypeRecognizer):
            raise TypeError
        if not isinstance(num_states, int):
            raise TypeError
        if not isinstance(spu, (int,float)):
            raise TypeError
        # Classifier factory
        factory = ClassifierFactory(type=type, num_states=num_states, spu=spu)
        # Create hmms
        hmms = {}
        for gesture_label in expressions.keys():
            hmms[gesture_label] = []
            for expression in expressions[gesture_label]:
                model, states = ModelExpression.createHmm(expression=expression, factory=factory)
                hmms[gesture_label].append(model)
        return hmms


    @staticmethod
    def createHmm(expression, factory):
        """
            starting from a deictic expression and a classifier factory, that function returns the hmm which implements
            the given expression.
        :param expression:
        :param factory:
        :return:
        """
        return factory.createClassifier(expression)

