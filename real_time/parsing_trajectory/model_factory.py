from gesture import *
from topology import *
import random

random.seed(0)

class Model():

    def __init__(self, n_states = 0, n_features = 0, name=None):
        # Model
        model = HiddenMarkovModel(name)
        # 3 states
        # distributions - {'A': 0.40, 'B': 0.40, 'O': 0.20}
        # a = DiscreteDistribution({'A': 0.795, 'B': 0.01, 'O': 0.195})
        # b = DiscreteDistribution({'A': 0.01, 'B': 0.795, 'O': 0.195})
        # o = DiscreteDistribution({'A': 0.495, 'B': 0.495, 'O': 0.01})
        # # states
        # s1 = State(a, name="A")
        # s2 = State(b, name="U")
        # s3 = State(o, name="V")
        # model.add_states(s1, s2, s3)
        #
        # # transitions
        # # start
        # model.add_transition(model.start, s1, 0.45)
        # model.add_transition(model.start, s2, 0.45)
        # model.add_transition(model.start, s3, 0.1)
        # # s1
        # model.add_transition(s1, s1, 0.4)
        # model.add_transition(s1, s2, 0.05)
        # model.add_transition(s1, s3, 0.4)
        # # s2
        # model.add_transition(s2, s1, 0.05)
        # model.add_transition(s2, s2, 0.5)
        # model.add_transition(s2, s3, 0.45)
        # # s3
        # model.add_transition(s3, s1, 0.45)
        # model.add_transition(s3, s2, 0.45)
        # model.add_transition(s3, s3, 0.1)
        # end

        # 6 states
        # # distributions - {'A': 0.40, 'U': 0.10, 'V': 0.10, 'Z': 0.10, 'X': 0.10, 'O': 0.2}
        # a = DiscreteDistribution({'A': 0.50, 'U': 0.0, 'V': 0.0, 'Z': 0.0, 'X': 0.0, 'O': 0.5})
        # u = DiscreteDistribution({'A': 0.0, 'U': 0.20, 'V': 0.10, 'Z': 0.10, 'X': 0.10, 'O': 0.5})
        # v = DiscreteDistribution({'A': 0.0, 'U': 0.10, 'V': 0.20, 'Z': 0.10, 'X': 0.10, 'O': 0.5})
        # z = DiscreteDistribution({'A': 0.0, 'U': 0.10, 'V': 0.10, 'Z': 0.20, 'X': 0.10, 'O': 0.5})
        # x = DiscreteDistribution({'A': 0.0, 'U': 0.10, 'V': 0.10, 'Z': 0.10, 'X': 0.20, 'O': 0.5})
        # o = DiscreteDistribution({'A': 0.40, 'U': 0.10, 'V': 0.10, 'Z': 0.10, 'X': 0.10, 'O': 0.2})
        #
        # # states
        # s1 = State(a, name="A")
        # s2 = State(u, name="U")
        # s3 = State(v, name="V")
        # s4 = State(z, name="Z")
        # s5 = State(x, name="X")
        # s6 = State(o, name="O")
        #
        # model.add_states(s1, s2, s3, s4, s5, s6)
        #
        # # transitions
        # d = 1/ 8
        # # start
        # model.add_transition(model.start, s1, 0.2)
        # model.add_transition(model.start, s2, 0.2)
        # model.add_transition(model.start, s3, 0.2)
        # model.add_transition(model.start, s4, 0.2)
        # model.add_transition(model.start, s5, 0.2)
        # # s1
        # model.add_transition(s1, s1, 0.5)
        # # s2
        # model.add_transition(s2, s2, 0.2)
        # model.add_transition(s2, s3, 0.2)
        # model.add_transition(s2, s4, 0.2)
        # model.add_transition(s2, s5, 0.2)
        # model.add_transition(s2, s6, 0.2)
        # # s3
        # model.add_transition(s3, s2, 0.2)
        # model.add_transition(s3, s3, 0.2)
        # model.add_transition(s3, s4, 0.2)
        # model.add_transition(s3, s5, 0.2)
        # model.add_transition(s3, s6, 0.2)
        # # s4
        # model.add_transition(s4, s2, 0.2)
        # model.add_transition(s4, s3, 0.2)
        # model.add_transition(s4, s4, 0.2)
        # model.add_transition(s4, s5, 0.2)
        # model.add_transition(s4, s6, 0.2)
        # # s5
        # model.add_transition(s5, s2, 0.2)
        # model.add_transition(s5, s3, 0.2)
        # model.add_transition(s5, s4, 0.2)
        # model.add_transition(s5, s5, 0.2)
        # model.add_transition(s5, s6, 0.2)
        # # s6
        # model.add_transition(s6, s1, 1/6)
        # model.add_transition(s6, s2, 1/6)
        # model.add_transition(s6, s3, 1/6)
        # model.add_transition(s6, s4, 1/6)
        # model.add_transition(s6, s5, 1/6)
        # model.add_transition(s6, s6, 1/6)
        # model.add_transition(s1, s6, 0.5)

        # Ergodic
        # states = self.__fix_parameters(name, n_states, [], [], model);
        # p_tr = 1 / (n_states + 2)
        # for i in range(0, n_states):
        #     model.add_transition(model.start, states[i], p_tr)
        #     model.add_transition(states[i], model.end, p_tr)
        #     for j in range(0, n_states):
        #         model.add_transition(states[i], states[j], p_tr)

        # Forward
        states = self.__fix_parameters(name, n_states, [], [], model);
        for i in range(0, n_states - 1):
            model.add_transition(model.start, states[i], n_states/1)
            for j in range(0, n_states - 1):
                model.add_transition(states[i], states[j], n_states/1)

        model.bake()
        self.model = model

    def train(self, samples):
        """

        :param samples:
        :return:
        """
        # Check parameters
        # Train
        self.model.fit(samples,use_pseudocount=True)

    def plot_sample(self, n_samples = 1):
        for i in range(0, n_samples):
            sequence = self.model.sample()
            print(sequence)

    def getModel(self):
        return self.model

    # Private Models #
    def __fix_parameters(self, name, n_States, emissions, state_names, model):
        # default init of missing emission distributions - {'A': 0.40, 'B': 0.40, 'O': 0.20} - {'A':0.40, 'U':0.10, 'V':0.10, 'Z':0.10, 'X':0.10, 'O':0.2}
        for i in range(len(emissions), n_States):
            # 3 caratteri
            #distribution_values = numpy.random.dirichlet(numpy.ones(3), size=1)[0]
            #emissions.append(DiscreteDistribution({'A': distribution_values[0], 'B':distribution_values[1], 'O':distribution_values[2]}))
            # 6 caratteri
            distribution_values = numpy.random.dirichlet(numpy.ones(6), size=1)
            emissions.append(DiscreteDistribution({'A':distribution_values[0][0], 'B':distribution_values[0][1],
                                                   'C':distribution_values[0][2], 'D':distribution_values[0][3],
                                                   'E':distribution_values[0][4], 'O':distribution_values[0][5]}))

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
