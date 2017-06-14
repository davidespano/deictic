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
        plt.text(j, i, "{0:.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
class_names = ['T', 'N', 'D', 'P', 'X', 'H', 'I ', '! ',

               ''+ chr(0x2205)+ ' ', #null
               '' + chr(0x279C)+ '', # arrow head
               '' + chr(0x03C8) + '', # pitchfork
                '  ' + chr(0x2721) + '', # six point star
               '' +chr(0x2731),  # asterisk
               '' + chr(0x2669) #half note
               ];
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


)



np.set_printoptions(precision=2)

# delete_mark 2A32

# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names,
#                      title='Confusion matrix, without normalization')


# Plot multistroke matrix
plt.figure()
plot_confusion_matrix(deictic_unistroke, classes=class_names, normalize=True,
                      title='DEICTIC', cmap=plt.cm.Greys)
plt.figure()
plot_confusion_matrix(ad_hoc_multistroke, classes=class_names, normalize=True,
                      title='Ad-hoc HMMs', cmap=plt.cm.Greys)

plt.show()