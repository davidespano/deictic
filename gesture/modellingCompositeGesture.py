from dataset import *
from pomegranate import *
from topology import *
import matplotlib.pyplot as plt
import random
from enum import Enum
from gesture import *

######### Modelled Composite Gesture #########
## create_sequence
# Links the input hmms for recognizing a sequence
def create_sequence(models):
    name = ''
    seq = []
    for model in models:
        seq.append(model)
        if name == '':
            name = model.name
        else:
            name = name+'_'+model.name

    sequence, seq_edges = HiddenMarkovModelTopology.sequence(seq);
    sequence.name = name
    return sequence, seq_edges

## Iterative
# Links the input hmms for recognizing an iteration
def create_iterative(model):
    iterative, seq_edges = HiddenMarkovModelTopology.iterative(model)
    iterative.name = model.name
    return iterative, seq_edges

## Choice
# Links the input hmms to build choice operator
def create_choice(models):
    # Componenti
    seq = []
    for model in models:
        seq.append(model)

    # Crea modello
    choice, seq_edges = HiddenMarkovModelTopology.choice(seq)
    return choice, seq_edges

## Disabling
# Links the input hmms to build disabling operator
def create_disabling(first_model, second_model, seq_first):
    # Crea modello
    disabling, seq_edges = HiddenMarkovModelTopology.disabling(first_model, second_model, seq_first)
    disabling.name = first_model.name + '_' + second_model.name
    return disabling, seq_edges

## Parallel
# Links the input hmms to build parallel operator
def create_parallel(first_model, second_model):
    # Crea modello
    parallel, seq_edges = HiddenMarkovModelTopology.parallel(first_model, second_model);
    parallel.name = first_model.name + '_' + second_model.name
    return parallel, seq_edges