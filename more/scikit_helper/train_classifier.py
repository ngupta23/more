import numpy as np
import matplotlib.pyplot as plt

from .common import time_usage
from .plot_classification import print_classification_details
from sklearn import metrics as mt


def train_classifier(estimator, X, y, cv, verbose=False):
    """
    Manual training of Classifiers using loops (does not parallelize)
    Not recommended for detailed analysis (use Scikit Learn's internal
    functions with njobs for parallelism). Only useful whe you want to
    vizualize the confusion matrix and the classification report
    """

    # For ROC
    printROC = False  # ROC is not available to end users for now.
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for iter_num, (train_indices, test_indices) in enumerate(cv.split(X, y)):
        with time_usage(" Classifier: Iteration " + str(iter_num+1)):
            estimator.fit(X[train_indices], y[train_indices])  # train object
            # get test set precitions
            y_hat = estimator.predict(X[test_indices])

            if (printROC is True):
                print_classification_details(
                    actual=y[test_indices], predicted=y_hat, verbose=False)
            else:
                print_classification_details(
                    actual=y[test_indices], predicted=y_hat, verbose=verbose)

            if (printROC is True):
                # For ROC
                probas_ = estimator.predict_proba(X[test_indices])
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = mt.roc_curve(
                    y[test_indices], probas_[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = mt.auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=1, alpha=0.3,
                         label='ROC fold %d (AUC = %0.2f)' %
                         (iter_num, roc_auc))

    if (printROC is True):
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
                 color='r', label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = mt.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (
                     mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                         color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()
