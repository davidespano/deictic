from gesture import *
from topology import *
import random
import datetime
# copy
import copy

class ModelFactory():

    __chars = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7','O']# 'B0True','B1True','B2True','B3True', 'B0False','B1False','B2False','B3False', 'O']

    @staticmethod
    def ergodic(nome=None, n_states=0):
        model = HiddenMarkovModel(name)
        # Ergodic
        states = self.__fix_parameters(self.name, n_states, [], [], self.model);
        p_tr = 1 / (n_states + 2)
        for i in range(0, n_states):
            self.model.add_transition(self.model.start, states[i], p_tr)
            self.model.add_transition(states[i], self.model.end, p_tr)
            for j in range(0, n_states):
                self.model.add_transition(states[i], states[j], p_tr)
        # bake
        model.bake()
        return model

    @staticmethod
    def forward(name=None, n_states=0):
        model = HiddenMarkovModel(name)
        # Forward
        states = ModelFactory.__fix_parameters(name, n_states, [], [], model);
        for i in range(0, n_states - 1):
            model.add_transition(model.start, states[i], n_states / 1)
            for j in range(0, n_states - 1):
                model.add_transition(states[i], states[j], n_states / 1)
        # bake
        model.bake()
        return model

    @staticmethod
    def sequenceAlignment(name=None, ideal_sequence=None):
        model = HiddenMarkovModel(name)
        ### create distribution
        # Define the distribution for insertions
        i_d = DiscreteDistribution(
            {'A0': 0.1111, 'A1': 0.1111, 'A2': 0.1111, 'A3': 0.1111, 'A4': 0.1111,
             'A5': 0.1111, 'A6': 0.1111, 'A7': 0.1111, 'O': 0.1112}
        )
        s_d = {'A0': 0.00, 'A1': 0.00, 'A2': 0.00, 'A3': 0.00, 'A4': 0.00,
             'A5': 0.00, 'A6': 0.00, 'A7': 0.00, 'O': 0.00}

        ### create states
        # Create the insert states
        insert_states = []
        for index in range(len(ideal_sequence)+1):
            insert_states.append(State(i_d, name="I"+str(index)))
        # Create the match states
        match_states = []
        for index in range(len(ideal_sequence)):
            n_d = deepcopy(s_d)
            n_d[ideal_sequence[index]] = 1.0
            match_states.append(State((DiscreteDistribution(n_d)), name="M" + str(index + 1)))
        # Create the match and the delete states
        delete_states = []
        for index in range(len(ideal_sequence)):
            delete_states.append(State(None, name="D"+str(index+1)))
        # add states
        states = [state for state in insert_states]
        states = states + [state for state in match_states]
        states = states + [state for state in delete_states]
        model.add_states(states)

        ### create transitions
        # Create transitions from match states
        model.add_transition(model.start, match_states[0], 0.9)
        model.add_transition(model.start, insert_states[0], 0.1)
        for index_state in range(0, len(match_states)-1):
            model.add_transition(match_states[index_state], match_states[index_state+1], 0.9)
            model.add_transition(match_states[index_state], insert_states[index_state + 1], 0.05)
            model.add_transition(match_states[index_state], delete_states[index_state + 1], 0.05)
        model.add_transition(match_states[-1], model.end, 0.9)
        model.add_transition(match_states[-1], insert_states[-1], 0.1)
        # Create transitions from insert states
        for index_state in range(0, len(insert_states)-1):
            model.add_transition(insert_states[index_state], insert_states[index_state], 0.7)
            model.add_transition(insert_states[index_state], delete_states[index_state], 0.15)
            model.add_transition(insert_states[index_state], match_states[index_state], 0.15)
        model.add_transition(insert_states[-1], insert_states[-1], 0.85)
        model.add_transition(insert_states[-1], model.end, 0.15)
        # Create transitions from delete states
        for index_state in range(0, len(delete_states)-1):
            model.add_transition(delete_states[index_state], delete_states[index_state+1], 0.15)
            model.add_transition(delete_states[index_state], insert_states[index_state+1], 0.15)
            model.add_transition(delete_states[index_state], match_states[index_state+1], 0.70)
        model.add_transition(delete_states[-1], insert_states[-1], 0.30)
        model.add_transition(delete_states[-1], model.end, 0.70)

        # bake
        model.bake()
        return model


    @staticmethod
    def train(model, samples):
        # Check parameters
        # get sequence and train
        samples = (sample[0] for sample in samples)
        model.fit(samples, transition_pseudocount=10, use_pseudocount=True)

    @staticmethod
    def plot_sample(self, n_samples = 1):
        # check parameters
        # plot
        for i in range(0, n_samples):
            sequence = self.model.sample()
            print(sequence)

    # private methods #
    @staticmethod
    def __fix_parameters(name, n_States, emissions, state_names, model):
        for i in range(len(emissions), n_States):
            distribution_values = numpy.random.dirichlet(numpy.ones(len(ModelFactory.__chars)), size=1)[0]
            values = {}
            random.seed(datetime.datetime.now())
            for index in range(0, len(ModelFactory.__chars)):
                values[ModelFactory.__chars[index]] = distribution_values[index]
            emissions.append(DiscreteDistribution(values))

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
