import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics as mt


def print_classification_details(actual, predicted, verbose=False):
    # print the accuracy and confusion matrix
    cm = mt.confusion_matrix(actual, predicted)
    cr = mt.classification_report(actual, predicted)

    print("confusion matrix\n", cm)
    print(cr)

    if (verbose is True):
        plot_classification_report(cr)


def plot_classification_report(cr, title=None, cmap='RdBu'):
    """
    Adapted from
    https://medium.com/district-data-labs/visual-diagnostics-for-more-informed-machine-learning-7ec92960c96b
    """
    title = title or 'Classification report'
    lines = cr.split('\n')
    classes = []
    matrix = []

    for line in lines[2:(len(lines)-5)]:
        s = line.split()
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)

    fig, ax = plt.subplots(1)

    for column in range(len(matrix)+1):
        for row in range(len(classes)):
            txt = matrix[row][column]
            # ax.text(column,row,matrix[row][column],va='center',ha='center')
            ax.text(column, row, txt, va='center', ha='center',
                    size="x-large", bbox=dict(facecolor='white', alpha=0.5))

    fig = plt.imshow(matrix, interpolation='nearest',
                     cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(classes)+1)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['Precision', 'Recall', 'F1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()
