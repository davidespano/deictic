from pomegranate import *


class HiddenMarkovModelTopology :

    def ergodic(self, name='ergodic-model', n_States= 1, emissions = [], state_names = []):
        model = HiddenMarkovModel(name)

        states = self.__fix_parameters(name, n_States, emissions, state_names);

        #init transitions
        p_tr = 1 / (n_States + 2)
        for i in range(0, n_States):
            model.add_transition(model.start, states[i], p_tr)
            model.add_transition(states[i], model.end, p_tr)
            for j in range (0, n_States):
                model.add_transition(states[i], states[j], p_tr)

        model.bake()
        return model

    def left_right(self, name='leftRight-model', n_states = 1, emissions = [], state_names = []):

        model = HiddenMarkovModel(name)

        states = self.__fix_parameters(name, n_states, emissions, state_names, model);

        #init transitions
        for i in range(0, n_states - 1):
            for j in range(i, n_states):
                model.add_transition(states[i], states[j], 1 / (n_states - i))

        model.add_transition(model.start, states[0], 1)
        model.add_transition(states[n_states - 1], states[n_states - 1], 0.5)
        model.add_transition(states[n_states - 1], model.end, 0.5)

        model.bake()
        return model

    def forward(self, name="forward-model", n_states = 1, emissions = [], state_names = []):
        model = HiddenMarkovModel(name)



        states = self.__fix_parameters(name, n_states, emissions, state_names, model);
        for i in range(0, n_states - 1):
            model.add_transition(states[i], states[i], 0.5)
            model.add_transition(states[i], states[i+1], 0.5)

        model.add_transition(model.start, states[0], 1)
        model.add_transition(states[n_states - 1], states[n_states - 1], 0.5)
        model.add_transition(states[n_states - 1], model.end, 0.5)

        model.bake()

        return model

    @staticmethod
    def sequence(operands, gt_edges = None):
        if len(operands) == 0:
            return None

        if len(operands) == 1:
            return operands[0]

        # tracking edges added for implementing the sequence
        seq_edges = []
        if gt_edges is not None:
            seq_edges = gt_edges

        starts = []
        ends = []

        for i in range(0, len(operands)):
            starts.append(HiddenMarkovModelTopology.__find_start_states(operands[i]))
            ends.append(HiddenMarkovModelTopology.__find_end_states(operands[i]))

        for i in range(1, len(operands)):
            for j in range(0,len(ends[i-1])):
                for h in range(0, len(starts[i])):
                    seq_edges.append((ends[i-1][j], starts[i][h]))

        sequence = operands[0]
        for i in range(1,len(operands)):
            sequence.concatenate(operands[i])
        sequence.bake();

        return sequence, seq_edges

    @staticmethod
    def choice(operands, gt_edges = None):
        if len(operands) == 0:
            return None

        if len(operands) == 1:
            return operands[0]

        choice = operands[0];

        # graph union
        for i in range(1,len(operands)):
            choice.graph = networkx.union(choice.graph, operands[i].graph)

        # setting the transition from the starting state to all the operands
        for i in range(1, len(operands)):
            operand = operands[i]
            toRemove = []
            for edge in operand.graph.edges():
                if edge[0] == operand.start:
                    choice.add_transition(choice.start, edge[1], 1.0)
                    toRemove.append(edge)
            for edge in toRemove:
                operand.graph.remove_edge(edge[0], edge[1])

        # setting the transition to the final state for all the operands
        for i in range(1, len(operands)):
            operand = operands[i]
            toRemove = []
            for edge in operand.graph.edges():
                if edge[1] == operand.end:
                    choice.add_transition(edge[0], choice.end, 1.0)
                    toRemove.append(edge)
            for edge in toRemove:
                operand.graph.remove_edge(edge[0], edge[1])

        choice.bake()
        return choice, gt_edges

    @staticmethod
    def iterative(operand, gt_edges = None):
        if operand is None:
            return None

        seq_edges = []
        if gt_edges is not None:
            seq_edges = gt_edges

        start_states = []
        for edge in operand.graph.edges():
            if edge[0] == operand.start:
                start_states.append(edge[1])

        for edge in operand.graph.edges():
            if edge[1] == operand.end:
                prob = numpy.exp(operand.graph[edge[0]][edge[1]]['probability'])/(len(start_states) + 1)
                operand.graph[edge[0]][edge[1]]['probability'] = numpy.log(prob)
                operand.add_transition(edge[0], start_states[0], prob)

        operand.bake()
        return operand, seq_edges

    @staticmethod
    def disabling(disabling_term, right_operand, gt_edges = None):

        if disabling_term is None or right_operand is None:
            return None

        disabling_term.graph = networkx.union(disabling_term.graph, right_operand.graph)


        l_end = HiddenMarkovModelTopology.__find_end_states(disabling_term)
        r_start = HiddenMarkovModelTopology.__find_start_states(right_operand)
        r_end = HiddenMarkovModelTopology.__find_end_states(right_operand)

        seq_edges = []
        if gt_edges is not None:
            seq_edges[:] = gt_edges[:]

            # add a transition to the disabling term from all the ground terms in the left operand
            for edge in gt_edges:
                print('edge ({}, {})'.format(edge[0].name, edge[1].name))
                print(disabling_term.graph[edge[0]][edge[1]])
                prob = numpy.exp(disabling_term.graph[edge[0]][edge[1]]['probability'])
                #print('edge ({} {}) prob: {}, updated: {}'.format(edge[0].name, edge[1].name, prob, prob / (len(r_start) + 1)))
                prob = prob / (len(r_start) + 1)
                disabling_term.graph[edge[0]][edge[1]]['probability'] = numpy.log(prob)
                for i in range(0, len(r_start)):
                    disabling_term.add_transition(edge[0], r_start[i], prob)
                    seq_edges.append((edge[0], r_start[i]))

        # add a transition to the disabling term for all terms that end the left operand
        for i in range(0, len(l_end)):
            prob = numpy.exp(disabling_term.graph[l_end[i]][disabling_term.end]['probability']) / len(r_start)
            for j in range(0, len(r_start)):
                disabling_term.add_transition(l_end[i], r_start[j], prob)
                seq_edges.append((l_end[i], r_start[i]))
            disabling_term.graph.remove_edge(l_end[i], disabling_term.end)



        # add a transition from all final states in the right operand to the end state
        for i in range(0, len(r_end)):
            prob = numpy.exp(disabling_term.graph[r_end[i]][right_operand.end]['probability'])
            disabling_term.add_transition(r_end[i], disabling_term.end, prob)
        disabling_term.bake()
        return disabling_term, seq_edges

    @staticmethod
    def parallel(first, second, gt_edges = None):

        if first is None or second is None:
            return None

        par = HiddenMarkovModel("parallel")

        # tengo traccia degli accoppiamenti degli stati in una matrice
        # le righe per la prima hmm, le colonne per la seconda
        stateMatrix = [];

        for i in range(0, len(first.states)):
            s1 = first.states[i]
            stateMatrix.append([])
            if s1 != first.start and s1 != first.end:
                for j in range(0, len(second.states)):
                    s2 = second.states[j]

                    if s2 != second.start and s2 != second.end:
                        # recupero le distribuzioni delle due variabili
                        e1 = s1.distribution;
                        e2 = s2.distribution;

                        # la distribuzione dello stato parallelo s1,s2 e' una distribuzione
                        # congiunta a componenti indipendenti
                        e1e2 = IndependentComponentsDistribution(
                            [e1.distributions[0].copy(),
                             e1.distributions[1].copy(),
                             e2.distributions[0].copy(),
                             e2.distributions[1].copy()]);

                        # creo lo stato parallelo e lo aggiungo alla hmm
                        s1s2 = State(e1e2, s1.name + ", " + s2.name)
                        par.add_state(s1s2)
                        stateMatrix[i].append(s1s2)

        for i in range(0, len(stateMatrix)):
            for j in range(0, len(stateMatrix[i])):
                # i,j e' uno stato della hmm parallela. Lo confronto con una altro generico stato
                # della stessa macchina
                par_sf = stateMatrix[i][j]
                for h in range(0, len(stateMatrix)):
                    for k in range(0, len(stateMatrix[h])):
                        # h, k e' un altro stato della hmm parallela. Aggiungo una transizione
                        # solo nel caso ci sia una transizione da i ad h nella prima hmm
                        # ed una transizione da j a k nella seconda
                        edge_ih = first.graph.has_edge(first.states[i], first.states[h])
                        edge_jk = second.graph.has_edge(second.states[j], second.states[k])

                        if edge_ih  and edge_jk :
                            # recupero gli stati della hmm parallela

                            par_ss = stateMatrix[h][k]

                            # la probabilita' di transizione e' il prodotto delle due transizioni
                            # visto che sono indipendenti

                            l = numpy.exp( first.graph.edge[first.states[i]][first.states[h]]['probability'] +
                                           second.graph.edge[second.states[j]][second.states[k]]['probability'])

                            par.add_transition(par_sf, par_ss, l)

                # inserisco le transizioni di start e di end
                if first.graph.has_edge(first.start, first.states[i]) and second.graph.has_edge(second.start, second.states[j]):
                    l = numpy.exp( first.graph.edge[first.start][first.states[i]]['probability'] +
                                    second.graph.edge[second.start][second.states[j]]['probability'])
                    par.add_transition(par.start, par_sf, l)

                if first.graph.has_edge(first.states[i], first.end) and second.graph.has_edge(second.states[j], second.end):
                    l = numpy.exp(first.graph.edge[first.states[i]][first.end]['probability'] +
                                  second.graph.edge[second.states[j]][second.end]['probability'])
                    par.add_transition(par_sf, par.end, l)


        par.bake()
        return par, gt_edges




    def __fix_parameters(self, name, n_States, emissions, state_names, model):
        # default init of missing emission distributions
        for i in range(len(emissions), n_States):
            emissions.append(NormalDistribution(0, 1.0))


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

    @staticmethod
    def __find_start_states(term):
        start = []
        for edge in term.graph.edges():
            if edge[0] == term.start:
                start.append(edge[1])
        return start

    @staticmethod
    def __find_end_states(term):
        end = []
        for edge in term.graph.edges():
            if edge[1] == term.end:
                end.append(edge[0])
        return end