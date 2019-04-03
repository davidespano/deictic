# pomegranate
import pomegranate


class DistributionFactory():
    '''
        This class, given the recognizer type and the number of states, creates the distribution list, necessary
        in the creation of hmm models.
    '''

    # public methods
    def __init__(self, type=None):
        """
        :param type: specifies the recognizer type (parsed or offine) and, consequently, the distribution type.
        :param chars: at the moment is used only for the parsed recognizer.
        """
        self.__function = {
            TypeRecognizer.offline: self.__NormalDistribution,
            TypeRecognizer.online: self.__DiscreteDistribution,
        }
        self.__type = type
        self.__chars = chars
    def getEmissions(self, *args):
        """
            given the number of states, 'getEmissions' returns a set of distributions generated randomly.
        :param num_states: the number of states which will compose the hidden markov model
        :param emissions:
        :return: the set of distributions
        """
        return self.__function[self.__type](*args)

    # private methods
    def __DiscreteDistribution(self, typeOperator=None, num_states=0, emissions=[]):
        for i in range(0, num_states):
            random.seed(datetime.datetime.now())
            distribution_values = numpy.random.dirichlet(numpy.ones(len(self.__chars)), size=1)[0]
            emissions.append({self.__chars[index]: distribution_values[index] for index in range(0, len(self.__chars))})
        return emissions

    def __NormalDistribution(self, typeOperator=None, num_states=0, emissions=[]):
        distributions = []
        if typeOperator == OpEnum.Arc:
            return None
        elif typeOperator == OpEnum.Line:
            step_x = dx / max(samples - 1, 1)
            step_y = dy / max(samples - 1, 1)
            for i in range(0, samples):
                a = (startPoint[0] + (i * step_x)) * scale
                b = (startPoint[1] + (i * step_y)) * scale
                gaussianX = NormalDistribution(a, self.scale * 0.01)
                gaussianY = NormalDistribution(b, self.scale * 0.01)
                distributions.append(IndependentComponentsDistribution([gaussianX, gaussianY]))