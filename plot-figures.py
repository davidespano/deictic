print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt



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
            plt.text(j, i, "{0:.2f}".format(cm[i, j]),
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
    print(correct)
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

# synthetic unistroke

synth_unistroke_iterative = np.matrix(
    [[330,	0,	 0,	  0,	0,	 0,	  0],
     [0,	330, 0,	  0,	0,	 0,	  0],
     [0,	1,	 329, 0,	0,	 0,	  0],
     [3,	0,	 0,	  327,	0,	 0,	  0],
#     [0,	0,	 0,	  0,	330, 0,	  0],
     [0,	1,	 0,	  0,	0,	 329, 0],
     [0,	2,	 0,	  0,	15,	 0,	  313]]
)

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
     [0,	0,	    0,	   0,	0,	   0,	330]]
)

synth_unistroke_sequence_names = [
    'V » ' + chr(0x232B), # v + delete
    chr(0x25B3) + ' » ' + chr(0x25CB),  # triangle + rectangle
    chr(0x25CB) + ' » ' + chr(0x2113), # circle + pigtail
    chr(0x22C0) + ' » ' + chr(0x279C), # caret + arrow
    '[' + ' » ' + chr(0x2605), # left sq bracket + star
    '?' + ' » ' + '[ ', # question mark + left sq bracket
    chr(0x2713) + ' » ' + ']'
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

np.set_printoptions(precision=2)

# delete 2A32

# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names,
#                      title='Confusion matrix, without normalization')


# Plot multistroke matrix
plt.figure()
plot_confusion_matrix(synth_unistroke_sequence, classes=synth_unistroke_sequence_names, normalize=True,
                       title='Iterative', cmap=plt.cm.Greys)
#plt.figure()
#plot_confusion_matrix(ad_hoc_unistroke, classes=uni_class_names, normalize=True,
#                       title='Ad-hoc HMMs', cmap=plt.cm.Greys)

plt.show()

#accuracy(ad_hoc_multistroke, 600);
