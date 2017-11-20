from gesture import *
from topology import *
import random

random.seed(0)

class Model():

    def __init__(self, n_states = 0, n_features = 0, name=None):

        # Model
        model = HiddenMarkovModel(name)

        states = self.__fix_parameters(name, n_states, [], [], model);
        for i in range(0, n_states - 1):
            model.add_transition(states[i], states[i], 0.5)
            model.add_transition(states[i], states[i + 1], 0.5)

        model.add_transition(model.start, states[0], 1)
        model.add_transition(states[n_states - 1], states[n_states - 1], 0.5)
        model.add_transition(states[n_states - 1], model.end, 0.5)
        model.bake()
        self.model = model

    def train(self, samples):
        """

        :param samples:
        :return:
        """
        # Check parameters
        # Train
        for sample in samples:
            self.model.fit(sample, use_pseudocount=True)

    def plot_sample(self, n_samples = 1):
        for i in range(1, n_samples):
            sequence = self.model.sample()
            print(sequence)

    def getModel(self):
        return self.model

    # Private Models #
    def __fix_parameters(self, name, n_States, emissions, state_names, model):
        # default init of missing emission distributions
        for i in range(len(emissions), n_States):
            emissions.append(DiscreteDistribution({'A0': 0.40, 'B0': 0.10, 'B1': 0.10, 'B2': 0.10, 'B3': 0.10, 'O0': 0.2}))
        # default init of missing state names
        cl_state = state_names[:]
        for i in range(len(state_names), n_States):
            cl_state.append(name + '-state-' + str(i))
        # init states
        states = []
        for i in range(0, n_States):
            states.append(State(emissions[i], cl_state[i]))
            model.add_state(states[i])
        return states


########## Emissions #########
## gesture_emissions
# Defines the emissions for the hmm
def gesture_emissions(n_states,  n_features = 2, scale = 1, weights = None):
    distributions = []
    step = scale / n_states;
    for i in range(0, n_states):
        distr_list = []
        for i in range(0, n_features):
            value = scale - (i * step)
            gaussian = NormalDistribution(value, scale * 0.01)
            distr_list.append(gaussian)
        if weights is None:
            distributions.append(IndependentComponentsDistribution(distr_list))
        else:
            distributions.append(IndependentComponentsDistribution(distr_list, weights = weights))
    return  distributions

## create_hmm_gesture
# Defines the hmm for recognizing the specified gesture
def create_hmm_gesture(name, training_set, n_states = 8, n_features = 2, weights = None, stroke= 1):
    # Creates hmm
    topology_factory = HiddenMarkovModelTopology()

    #TODO eliminare questo orrore
    emissions = gesture_emissions(n_states, n_features = 2, scale=100, weights=weights)
    model = topology_factory.forward(name, n_states, emissions)
    # Train
    model.fit(training_set)

    sps = n_states / stroke
    if n_features == 3:
        for i in range(0, len(model.states)):
            state = model.states[i]
            if not state.distribution is None:
                x = state.distribution.distributions[0]
                y = state.distribution.distributions[1]
                # s = NormalDistribution(self.stroke + 1.0, (i +1 ) * step)
                s = NormalDistribution(math.trunc(i/sps) + 1.0, 0.01)
                state.distribution = IndependentComponentsDistribution([x, y, s], weights=[1, 1, 10000])
    return model

## plot_gesture
# Plots the model input examples.
#@staticmethod
# Plot a model gesture
def plot_gesture(model):
    for i in range(1, 3):
        sequence = model.sample()
        result = numpy.array(sequence).astype('float')
        plt.axis("equal")
        plt.plot(result[:, 0], result[:, 1], marker=".")
        plt.show()
