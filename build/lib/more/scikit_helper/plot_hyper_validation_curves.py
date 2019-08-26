import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve


###############################
# Grid Search Hyperparameters #
###############################

# Not loaded yet
# If you want to release this, include this in the __init__.py file,
# and show an example of how to use.

def plot_hyper_validation_curve(estimator,
                                X,
                                y,
                                param_name,
                                param_range,
                                title="Validation Curve",
                                xlabel=None,
                                legend_loc='best',
                                logX=False,
                                cv=None,
                                scoring="accuracy",
                                n_jobs=None,
                                verbose=0,
                                ax=None):
    """
    * Adapted from:
    https://chrisalbon.com/machine_learning/model_evaluation/plot_the_validation_curve/
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
    """

    # Calculate accuracy on training and test set using range
    # of parameter values
    train_scores, test_scores = validation_curve(estimator=estimator,
                                                 X=X,
                                                 y=y,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring=scoring,
                                                 verbose=verbose,
                                                 n_jobs=n_jobs
                                                 )

    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Clean up for log X axis plotting
    x_label_append = ""  # empty if not log scale
    if (logX is True):
        param_range = np.log10(param_range)
        x_label_append = " (log scale)"

    # Plot mean accuracy scores for training and test sets
    ax.plot(param_range, train_mean, label="Training Score", color="r")
    ax.plot(param_range, test_mean, label="Cross-Validation Score", color="g")

    # Plot accurancy bands for training and test sets
    ax.fill_between(param_range, train_mean - train_std,
                    train_mean + train_std, color="r", alpha=0.1)
    ax.fill_between(param_range, test_mean - test_std,
                    test_mean + test_std, color="g", alpha=0.1)

    # Add annotations
    ax.title.set_text(title)
    ax.set_ylabel((scoring + " Score").title())
    if xlabel is None:
        ax.set_xlabel("Hyperparameter: " + param_name + x_label_append)
    else:
        ax.set_xlabel(xlabel + x_label_append)

    ax.legend(loc=legend_loc)


def plot_hyper_validation_curves(estimator,
                                 X,
                                 y,
                                 param_grid,
                                 scoring,
                                 logX=False,
                                 cv=None,
                                 refit='Accuracy',
                                 n_jobs=None,
                                 verbose=0,
                                 arFigsize=None):
    """
    * Adapted from
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
    * Uses GridSearchCV to evaluate many metrics at once. Hence, if evaluating
      many metrics at once, this is often faster than using validation_curve
      which will have to be run (trained) once time for each metric)
    * Currently supports only one hyperparameter at a time, but can
      score multiple metrics against that hyperparameter.
    """

    gs = GridSearchCV(estimator=estimator,
                      param_grid=param_grid,
                      scoring=scoring,
                      cv=cv,
                      refit=refit,
                      return_train_score=True,
                      n_jobs=-1,
                      verbose=verbose)
    gs.fit(X, y)
    results = gs.cv_results_

    # Plotting the result

    num_rows = math.ceil(len(scoring)/2)
    if (arFigsize is None):
        arFigsize = (12, num_rows*4)
    fig, axes = plt.subplots(num_rows, 2, figsize=arFigsize, squeeze=False)

    # For this function, we only expect 1 key
    # If you have more than 1 parameter for GC,
    # then results may not be as expected
    for hyper in param_grid:
        # Get the regular numpy array from the MaskedArray
        X_axis = np.array(results['param_' + hyper].data, dtype=float)

        x_label_append = " "
        if (logX is True):
            X_axis = np.log10(X_axis)
            x_label_append = " (log scale)"

        i = 0
        for scorer in scoring:
            title = "Validation Curve: " + scorer.title()

            ax = axes[math.floor(i/2), i % 2]
            for sample, style, color in (('train', 'o-', 'r'),
                                         ('test', 'o-', 'g')):
                sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
                sample_score_std = results['std_%s_%s' % (sample, scorer)]

                ax.plot(X_axis, sample_score_mean, style, color=color,
                        label="%s (%s)" % (scorer, sample))

                ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                                sample_score_mean + sample_score_std,
                                alpha=0.1 if sample == 'test' else 0.1,
                                color=color)

                # Add annotations
                ax.title.set_text(title)
                ax.set_ylim(-0.1, 1.1)
                ax.set_ylabel(scorer.title() + " Score")
                ax.set_xlabel("Hyperparameter: " + hyper + x_label_append)
                ax.grid()
                ax.legend(loc="best")

            best_index = np.nonzero(
                results['rank_test_%s' % scorer] == 1)[0][0]
            best_score = results['mean_test_%s' % scorer][best_index]

            # Plot a dotted vertical line at the best score for that scorer
            # marked by x
            ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                    linestyle='-.', color='black', marker='x',
                    markeredgewidth=3, ms=8)

            # Annotate the best score for that scorer
            ax.annotate("%0.2f" % best_score,
                        (X_axis[best_index], best_score + 0.005))

            i = i+1  # Increment for each scorer
    # return(fig)
