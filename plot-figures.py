print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt


class MNames:
    Null = chr(0x2205)
    ArrowHead = chr(0x279C)
    PitchFork = chr(0x03C8)
    SixPointStar = chr(0x2721)
    Asterisk =  chr(0x2731)
    HalfNote = chr(0x2669)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        den = cm.sum(axis=1)[:, np.newaxis]
        print(den)
        cm = cm.astype('float') / den[0]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if(cm[i,j] >= 0.01):
            plt.text(j, i, cm[i,j],
                     #"{0:.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def accuracy(matrix, max_val):

    correct = np.zeros(len(matrix));
    error = np.zeros(len(matrix));
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix)):
           if i == j:
               correct[i] = matrix[i,j]
           else:
                error[i] += matrix[i,j]
    print(correct / max_val)
    print(error)
    print("Accuracy mean: {0}, sd {1}".format(np.mean(correct, axis = 0)/max_val, np.std(correct, axis = 0)/max_val))
    print("Error mean: {0}, sd {1}".format(np.mean(error)/max_val, np.std(error)/max_val))

deictic_multistroke = np.matrix(
[[592,   0,   0,   0,   4,   0,   0,   0,   0,   0,   4,   0,   0,   0],
 [  0, 573,   0,   0,  12,  11,   0,   0,   0,   0,   0,   0,   4,   0],
 [  1,   0, 538,   0,  54,   0,   0,   1,   0,   0,   2,   4,   0,   0],
 [  1,   0,   4, 550,  34,   0,   0,   3,   0,   0,   5,   1,   0,   2],
 [  2,   0,   0,   0, 596,   0,   0,   1,   0,   0,   1,   0,   0,   0],
 [  0,  18,   0,   0,   0, 573,   1,   0,   0,   0,   0,   0,   8,   0],
 [  0,   0,   0,   0,   0,   2, 596,   0,   0,   0,   0,   0,   2,   0],
 [  1,   0,   0,   0,  33,   0,   0, 559,   0,   0,   3,   0,   0,   4],
 [  0,   0,   0,   0,  61,   0,   0,   3, 528,   0,   6,   1,   0,   0],
 [  0,   0,   0,   0,  51,   0,   0,   6,   0, 539,   4,   0,   0,   0],
 [  0,   0,   0,   0,   5,   0,   0,   0,   4,  14, 576,   0,   0,   1],
 [  0,   0,   0,   0,  29,   0,   0,   2,   1,   1,   1, 562,   0,   4],
 [  0,   0,   0,   0,   2,   0,   1,   0,   0,   0,   0,   0, 596,   0],
 [  1,   0,   0,   0,  58,   0,   0,   4,   6,   4,   1,   3,   0, 523]]
)

ad_hoc_multistroke = np.matrix(
[[ 557,    0,    0,    0,    0,    0,   30,    0,    0,    0,    3,    0,    0,    0],
 [   0,  589,    0,    0,    0,   11,    0,    0,    0,    0,    0,    0,    0,    0],
 [   0,    0,  334,  168,    0,    0,   10,    0,    0,    0,    0,    0,    0,   78],
 [   0,    0,  155,  408,    0,    0,    0,    0,    0,    0,    0,    0,    0,   27],
 [   0,    0,   10,    0,  545,    1,   24,    0,    0,    0,    0,    0,    0,    0],
 [   0,    0,    0,    0,    0,  590,    0,    0,    0,    0,    0,    0,    0,    0],
 [   0,    0,    0,    0,    0,    0,  580,    0,    0,    0,    0,    0,    0,    0],
 [   0,    0,    0,   13,    0,    0,    0,  584,    0,    0,    0,    0,    0,    3],
 [   0,    0,    4,    0,   10,    0,    0,    0,  553,    0,    0,    0,    0,   13],
 [   0,    0,    0,   12,   10,    2,    1,    0,    0,  477,    0,   40,    0,   48],
 [   1,    0,    0,    0,   20,   48,   24,   24,    0,    0,  460,    0,    0,   13],
 [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  550,    0,   10],
 [   0,    0,    0,    0,   22,    0,   23,    0,    0,    0,    0,    0,  537,    8],
 [   0,    0,    0,    0,    0,    5,   10,    7,    0,   10,    0,    3,    0,  545]]
)

deictic_unistroke = np.matrix(
[[320,   0,   0,  10,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0, 329,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0],
 [  6,   0, 302,  22,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0, 329,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0, 313,   0,   0,   1,   0,   0,  16,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0, 321,   0,   9,   0,   0,   0,   0,   0,   0,   0,   0],
 [  0,   1,   0,   0,   0,   1, 324,   2,   0,   0,   0,   0,   0,   2,   0,   0],
 [  0,   3,   0,   0,   1,   8,   0, 310,   0,   0,   0,   0,   0,   0,   0,   8],
 [  0,   0,   0,   1,   0,   0,   0,   0, 311,   0,   0,   0,  18,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   5,   0,   0, 309,   0,   0,   0,  16,   0,   0],
 [  0,   0,   0,   0,  12,   0,   0,   0,   0,   0, 318,   0,   0,   0,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 330,   0,   0,   0,   0],
 [  7,   0,   0,  18,   0,   0,   0,   0,   1,   0,   0,   0, 303,   1,   0,   0],
 [  0,   0,   0,   0,   0,   0,   5,   0,   0,   6,   0,   0,   6, 313,   0,   0],
 [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 330,   0],
 [  0,   0,   0,   0,   1,   8,   0,   0,   0,   0,   0,   0,   0,   0,   0, 321]]
)

ad_hoc_unistroke = np.matrix(
[[ 330,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,    0,    0,    0 ],
 [   0,  330,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,    0,    0,    0 ],
 [   3,    0,  326,    1,    0,    0,    0,    0,    0,    0,    0,    0,   0,    0,    0,    0 ],
 [   1,    0,    1,  327,    0,    0,    0,    0,    1,    0,    0,    0,   0,    0,    0,    0 ],
 [   0,    0,    0,    0,  324,    0,    0,    0,    0,    0,    6,    0,   0,    0,    0,    0 ],
 [   0,    0,    0,    0,    0,  322,    0,    8,    0,    0,    0,    0,   0,    0,    0,    0 ],
 [   0,    0,    0,    0,    3,    0,  327,    0,    0,    0,    0,    0,   0,    0,    0,    0 ],
 [   0,    0,    0,    0,    0,    0,    0,  330,    0,    0,    0,    0,   0,    0,    0,    0 ],
 [   0,    0,    0,    0,    0,    0,    0,    0,  328,    0,    0,    0,   2,    0,    0,    0 ],
 [   0,    0,    0,    0,    0,    0,    0,    0,    0,  329,    0,    0,   0,    1,    0,    0 ],
 [   0,    0,    0,    0,   13,    0,    0,    0,    0,    0,  317,    0,   0,    0,    0,    0 ],
 [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  330,   0,    0,    0,    0 ],
 [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 329,    1,    0,    0 ],
 [   0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    0,    0,   0,  328,    0,    0 ],
 [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,    0,  330,    0 ],
 [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   0,    0,    0,  330 ]]
)

# ------------------------------------------
# synthetic unistroke
# ------------------------------------------

synth_unistroke_iterative = np.matrix(
    [[330,	0,	 0,	  0,	0,	 0,	  0],
     [0,	330, 0,	  0,	0,	 0,	  0],
     [0,	1,	 329, 0,	0,	 0,	  0],
     [3,	0,	 0,	  327,	0,	 0,	  0],
     [0,	0,	 0,	  0,	330, 0,	  0],
     [0,	1,	 0,	  0,	0,	 329, 0],
     [0,	2,	 0,	  0,	15,	 0,	  313]]
)

synth_unistroke_iterative = np.delete(synth_unistroke_iterative, 4, axis=0)
synth_unistroke_iterative = np.delete(synth_unistroke_iterative, 4, axis=1)

synth_unistroke_iterative_names = [
    'V* ',
    '' + chr(0x25CB) + '* ',  # circle
    '[* ', # left square bracket
    '' + chr(0x2713) + '* ', # check
#    '' + chr(0x22C0) + '* ', # caret
    '' + chr(0x25B3) + '* ', # triangle
    '?* ', # question mark
]

synth_unistroke_sequence = np.matrix(
    [[330,	0,	    0,	   0,	0,	   0,	0],
     [0,	326,	4,	   0,	0,	   0,	0],
     [0,	0,	    330,   0,	0,	   0,	0],
     [0,	0,	    0,	   330,	0,	   0,	0],
     [0,	0,	    0,	   0,	330,   0,	0],
     [1,	0,	    0,	   0,	0,	   329,	0],
     [0,	0,	    0,	   0,	0,	   0 , 330]]
)

synth_unistroke_sequence = np.delete(synth_unistroke_sequence, 5, axis=0)
synth_unistroke_sequence = np.delete(synth_unistroke_sequence, 5, axis=1)

synth_unistroke_sequence_names = [
    'V » ' + chr(0x232B), # v + delete
    chr(0x25B3) + ' » ' + chr(0x2395),  # triangle + rectangle
    chr(0x25CB) + ' » ' + chr(0x2113), # circle + pigtail
    chr(0x22C0) + ' » ' + chr(0x279C), # caret + arrow
    '[' + ' » ' + chr(0x2605), # left sq bracket + star
    #'?' + ' » ' + '[ ', # question mark + left sq bracket
    chr(0x2713) + ' » ' + ']'
]

synth_unistroke_choice =  np.matrix([
    [316,   0,      0,      2,    2,    10,   0],
    [0,     330,    0,      0,    0,    0,    0],
    [0,     17,     313,    0,    0,    0,    0],
    [0,     7,      3,      315,  0,    0,    5],
    [2,     0,      0,      0,    328,  0,    0],
    [7,     0,      0,      0,    0,    323,  0],
    [0,     0,      0,      7,    1,    1,    321],
])

synth_unistroke_choice = np.delete(synth_unistroke_choice, 6, axis=0)
synth_unistroke_choice = np.delete(synth_unistroke_choice, 6, axis=1)


synth_unistroke_choice_names = [
    'V [] }',
    chr(0x25CB) + ' [] ' + chr(0x2605),  # circle [] star
    chr(0x25B3) + ' [] ' + chr(0x2395),  # triangle [] rectangle
    '{ [] ' + chr(0x22C0), # left curly brace [] caret
    '? [] X',               # question mark [] X
    chr(0x232B) + ' [] ' + chr(0x2713), # delete [] check
    #chr(0x279C) + ' [] ' +  chr(0x2113) + ' ' # arrow [] pigtail
]

synth_unistroke_parallel = np.matrix ([
    [330,	0,	    0,  	0,	    0,	    0,      0],
    [0,	    330,    0,	    0,	    0,	    0,	    0],
    [0,	    0,	    330,	0,	    0,	    0,	    0],
    [0,	    0,	    0,	    330,	0,	    0,	    0],
    [0,	    0,	    0,	    0,	    316,	0,	    14],
    [0,	    0,	    0,	    0,	    0,	    330,	0],
    [0,	    0,	    0,	    0,	    12,	    0,	    318]
])

synth_unistroke_parallel = np.delete(synth_unistroke_parallel, 3, axis=0)
synth_unistroke_parallel = np.delete(synth_unistroke_parallel, 3, axis=1)

synth_unistroke_parallel_names = [
    chr(0x22C0) + ' || ' +  chr(0x279C),    # caret || arrow
    chr(0x25CB) + ' || ' +  chr(0x2113),    # circle || pigtail
    '[ || ' + chr(0x2605),                  # left sq bracket || star
    #'? || [',                               # question mark || left sq bracket
    chr(0x2713) + ' || ]',                  # check || right sq bracket
    chr(0x25B3) + ' || ' + chr(0x2395),     # triangle [] rectangle
    'V || ]',                               # V || right sq bracket
]





# ------------------------------------------
# synthetic multistroke
# ------------------------------------------
synth_multistroke_iterative = np.matrix ([
    [598,	0,	    2,	    0,	    0],
    [0,	    600,	0,	    0,	    0],
    [12,	0,	    588,	0,	    0],
    [2,	    0,	    11,	    587,	0],
    [0,	    19,	    0,	    0,	    581]
])

synth_multistroke_iterative_names = [
    MNames.ArrowHead + '*',
    'N*',
    MNames.SixPointStar + '*',
    'D*',
    'I*'
]

synth_multistroke_sequence = np.matrix ([
    [593,   7,   0,   0,    0],
    [17,   583, 0,   0,    0],
    [3,   4,   566, 1,    25],
    [0,   1,   9,   588,  2],
    [0,   0,   2,   0,    598],
])

synth_multistroke_sequence_names = [
    MNames.PitchFork + ' + ' + '!',
    MNames.ArrowHead + ' + ' + MNames.HalfNote,
    MNames.Asterisk + ' + ' + MNames.SixPointStar,
    'D' +  ' + ' + 'N',
    'I' + ' + ' + 'T'
]



synth_multistroke_choice = np.matrix ([
    [571,	1,	    0,	    11,     17],
    [0,	    512,	14,	    28,	    46],
    [0,	    3,	    589,	2,	    5],
    [0,	    3,	    0,	    595,	2],
    [0,	    1,	    0,	    9,	    590]
])

synth_multistroke_choice_names = [
    MNames.HalfNote + " | " + 'P',
    '!' + " | " + 'H',
    MNames.Asterisk + " | " + 'N',
    MNames.ArrowHead + " | " + 'T',
    'D' + " | " + MNames.Null
]

synth_multistroke_parallel = np.matrix ([
    [596,   0,   1,   2,   0],
    [  0, 598,   0,   0,   2],
    [  0,   0, 599,   0,   0],
    [ 16,   0,   0, 584,   0],
    [  0,  15,   2,   0, 583],
])


synth_multistroke_parallel_names = [
    MNames.Asterisk + ''+ chr(0x00D7 )+ ' ' + 'N',
    'T' + ''+ chr(0x00D7 )+ ' ' + MNames.HalfNote,
    MNames.Null + ''+ chr(0x00D7 )+ ' ' + 'H',
    'I' + ''+ chr(0x00D7 )+ ' ' +  'D',
    MNames.SixPointStar + ''+ chr(0x00D7 )+ ' ' + 'P'
]

# Compute confusion matrix
multi_class_names = ['T', 'N', 'D', 'P', 'X', 'H', 'I ', '! ',

               ''+ chr(0x2205)+ ' ', #null
               '' + chr(0x279C)+ '', # arrow head
               '' + chr(0x03C8) + '', # pitchfork
                '  ' + chr(0x2721) + '', # six point star
               '' +chr(0x2731),  # asterisk
               '' + chr(0x2669) #half note
               ];

uni_class_names = [
    '' + chr(0x25B3) + ' ', # triangle
    'X ', # X
    '' + chr(0x2395) + ' ', # rectangle
    '' + chr(0x25CB) + ' ', # circle
    '' + chr(0x2713) + '', # check
    '' + chr(0x22C0) + '', # caret
    '? ', # question mark
    '' + chr(0x279C)+ ' ', # arrow
    '[ ', # left square bracket
    '] ', # right square bracket
    'V ', # V
    '' + chr(0x232B) + ' ', # delete
    '{ ', # left curly brace
    '} ', #right curly brace
    '' + chr(0x2605) +  ' ', # star
    '' + chr(0x2113) + ' ' # pigtail
];


iciap = np.matrix([
    [60, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 60, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 60, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 60, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 60, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 60, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 60, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 59, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 60, 0],
    [0, 0, 0, 0, 0, 0, 0, 0,  1, 59],

]);

iciap_names = [
    '' + chr(0x2190)+ ' ', # larrow
    '' + chr(0x2192)+ ' ', # rarrow
    'V ', # V
    '' + chr(0x22C0) + '', # caret
    '[ ', # left square bracket
    '] ', # right square bracket
    'X ', # X
    '' + chr(0x232B) + ' ', # delete
    '' + chr(0x25B3) + ' ', # triangle
    '' + chr(0x2395) + ' ', # rectangle
]


np.set_printoptions(precision=2)

# delete 2A32

# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names,
#                      title='Confusion matrix, without normalization')




def opPlot(matrix, names, title):
    plt.figure()
    for i in range(0,4):
        plt.subplot(221+i)
        plot_confusion_matrix(matrix[i], classes=names[i], normalize=True,
                          title=title[i], cmap=plt.cm.Greys)
    plt.show()


# Plot multistroke matrix
#plt.figure()
#plot_confusion_matrix(synth_unistroke_sequence, classes=synth_unistroke_sequence_names, normalize=True,
#                       title='Choice', cmap=plt.cm.Greys)
#plt.figure()
#plot_confusion_matrix(ad_hoc_unistroke, classes=uni_class_names, normalize=True,
#                       title='Ad-hoc HMMs', cmap=plt.cm.Greys)

#plt.show()

# opPlot(matrix = [synth_unistroke_iterative, synth_unistroke_sequence, synth_unistroke_choice, synth_unistroke_parallel],
#        names =  [synth_unistroke_iterative_names, synth_unistroke_sequence_names, synth_unistroke_choice_names, synth_unistroke_parallel_names],
#        title=   ["Iterative",'Sequence', 'Choice', 'Parallel'])

#opPlot(matrix = [synth_multistroke_iterative, synth_multistroke_sequence, synth_multistroke_choice, synth_multistroke_parallel],
#       names =  [synth_unistroke_iterative_names, synth_multistroke_sequence_names, synth_multistroke_choice_names, synth_multistroke_parallel_names],
#       title=   ["Iterative",'Sequence', 'Choice', 'Parallel'])

#accuracy(deictic_multistroke, 600);


plot_confusion_matrix(iciap, classes=iciap_names, normalize=False,
                       title='Confusion matrix', cmap=plt.cm.Greys)
plt.show()