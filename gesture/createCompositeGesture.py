from dataset import *
from pomegranate import *
from topology import *
import matplotlib.pyplot as plt
import random
from enum import Enum
from gesture import *


# Sequence
def create_sequence(models):
    name = ''
    seq = []
    for model in models:
        seq.append(model)
        if name == '':
            name = model.name
        else:
            name = name+'_'+model.name
    # Crea modello
    sequence, seq_edges = HiddenMarkovModelTopology.sequence(seq);
    sequence.name = name
    return sequence, seq_edges

# Iterative
def create_iterative(model):
    # Crea modello
    iterative, seq_edges = HiddenMarkovModelTopology.iterative(model)
    iterative.name = model.name
    return iterative, seq_edges

# Choice
def create_choice(models):
    # Componenti
    seq = []
    for model in models:
        seq.append(model)

    # Crea modello
    choice, seq_edges = HiddenMarkovModelTopology.choice(seq)
    return choice, seq_edges

# Disabling
def create_disabling(first_model, second_model, seq_first):
    # Crea modello
    disabling, seq_edges = HiddenMarkovModelTopology.disabling(first_model, second_model, seq_first)
    disabling.name = first_model.name + '_' + second_model.name
    return disabling, seq_edges

# Parallel
def create_parallel(first_model, second_model):
    # Crea modello
    parallel, seq_edges = HiddenMarkovModelTopology.parallel(first_model, second_model);
    parallel.name = first_model.name + '_' + second_model.name
    return parallel, seq_edges