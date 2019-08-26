import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import learning_curve
from .common import time_usage

###################
# Learning Curves #
###################


def plot_learning_curve(estimator,
                        title_suffix,
                        X,
                        y,
                        scoring="accuracy",
                        ylim=None,
                        cv=None,
                        n_jobs=None,
                        train_sizes=np.linspace(.2, 1.0, 5),
                        verbose=1,
                        ax=None):
    """
    NOTE:
        Adopted from:
            https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
            but allows plotting multiple metrics at a time when called from
            within plot_learning_curves (note the s at the end)

    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    train_sizes, train_scores, test_scores = learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            verbose=verbose
            )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    # Annotate Plots
    ax.title.set_text("Learning Curve: " + title_suffix)
    if (ylim is None):
        ax.set_ylim(-0.1, 1.1)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_ylabel(title_suffix + " Score")
    ax.set_xlabel("Training examples")
    ax.grid(b=True)
    ax.legend(loc="best")
    # return plt


def plot_learning_curves(estimator,
                         X,
                         y,
                         scoring, cv=None,
                         n_jobs=None,
                         verbose=1,
                         arFigsize=None):
    num_rows = math.ceil(len(scoring)/2)
    if (arFigsize is None):
        arFigsize = (12, num_rows*4)

    fig, axes = plt.subplots(num_rows, 2, figsize=arFigsize, squeeze=False)
    i = 0
    for scorer in scoring.keys():
        with time_usage(" Learning Curve | " + scorer):
            plot_learning_curve(estimator=estimator,
                                title_suffix=scorer.title(),
                                X=X,
                                y=y,
                                scoring=scoring[scorer],
                                cv=cv,
                                n_jobs=n_jobs,
                                verbose=verbose,
                                ax=axes[math.floor(i/2), i % 2]
                                )
            i = i+1

    # plt.show()
