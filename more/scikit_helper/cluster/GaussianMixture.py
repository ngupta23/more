import numpy as np
import matplotlib.pyplot as plt

import itertools
from sklearn import metrics as mt
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture


class GaussianMixtureHelper:
    """
    This code has been manipulated from the source available on sklearn's
    website documentation
    """

    def __init__(self,
                 X,
                 y=None,
                 n_components_range=range(2, 3),
                 cov_types=['spherical'],
                 metric='bic',
                 random_state=101):
        """
        Class to train and evaluate a Gaussian Mixture Cluster Model
        """
        self.X = X
        self.y = y
        self.n_components_range = n_components_range
        self.cov_types = cov_types

        # Add exception here is metric is not of the right type
        self.metric = metric
        self.random_state = random_state
        self.y_pred = None
        self.best_gmm = None
        self.best_gmm_bic = None
        self.best_gmm_aic = None
        self.bic = []
        self.aic = []
        self.lowest_bic = np.infty
        self.lowest_aic = np.infty

    def train(self):
        """
        Train the Gaissian Mixture Model across a range of cluster values
        and covariance types
        """
        for cov_type in self.cov_types:
            for n_components in self.n_components_range:
                # Fit a mixture of Gaussians with EM
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cov_type,
                                              random_state=self.random_state)
                gmm.fit(self.X)
                self.bic.append(gmm.bic(self.X))
                self.aic.append(gmm.aic(self.X))

                if self.bic[-1] < self.lowest_bic:
                    self.lowest_bic = self.bic[-1]
                    self.best_gmm_bic = gmm

                if self.aic[-1] < self.lowest_aic:
                    self.lowest_aic = self.aic[-1]
                    self.best_gmm_aic = gmm

        self.set_best_model()
        self.y_pred = self.predict(self.X)
        return(self)

    def set_metric(self, metric):
        self.metric = metric

    def set_best_model(self):
        """
        Use to set the best model to the one based on a specific metric
        Default Metric = 'bic'; Other Option(s): 'aic'
        """
        if (self.metric == 'bic'):
            self.best_gmm = self.best_gmm_bic
        elif(self.metric == 'aic'):
            self.best_gmm = self.best_gmm_aic

    def get_best_model(self):
        return(self.best_gmm)

    def plot_metrics(self, figsize=(12, 4)):
        """
        This code has been manipulated from the source available on
        sklearn's website documentation
        """

        plt.figure(figsize=figsize)

        # Plot the BIC scores
        spl = plt.subplot(1, 2, 1)
        color_iter = itertools.cycle(['k', 'r', 'b', 'g', 'c', 'm', 'y'])
        bars = []
        self.bic = np.array(self.bic)

        for i, (self.cov_type, color) in enumerate(zip(self.cov_types,
                                                       color_iter)):
            xpos = np.array(self.n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos,
                                self.bic[i*len(self.n_components_range):(i + 1)
                                         * len(self.n_components_range)],
                                width=.2, color=color))
        plt.xticks(self.n_components_range)
        plt.ylim([self.bic.min() * 1.01 - .01 * self.bic.max(),
                  self.bic.max()])
        plt.title('BIC score per model')

        xpos = np.min(self.n_components_range) - 0.4
        + np.mod(self.bic.argmin(), len(self.n_components_range))
        +  .2 * np.floor(self.bic.argmin() / len(self.n_components_range))

        plt.text(xpos, self.bic.min() * 0.97 + .03 * self.bic.max(),
                 '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], self.cov_types)

        # Plot the AIC scores
        spl = plt.subplot(1, 2, 2)
        color_iter = itertools.cycle(['k', 'r', 'b', 'g', 'c', 'm', 'y'])
        bars = []
        self.aic = np.array(self.aic)

        for i, (self.cov_type, color) in enumerate(zip(self.cov_types,
                                                       color_iter)):
            xpos = np.array(self.n_components_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos,
                                self.aic[
                                    i * len(self.n_components_range):
                                    (i + 1) * len(self.n_components_range)],
                                width=.2, color=color))
        plt.xticks(self.n_components_range)
        plt.ylim([self.aic.min() * 1.01 - .01 * self.aic.max(),
                  self.aic.max()])
        plt.title('AIC score per model')

        xpos = np.min(self.n_components_range) - 0.4
        + np.mod(self.aic.argmin(), len(self.n_components_range))
        + .2 * np.floor(self.aic.argmin() / len(self.n_components_range))

        plt.text(xpos, self.aic.min() * 0.97 + .03 * self.aic.max(),
                 '*', fontsize=14)
        spl.set_xlabel('Number of components')
        spl.legend([b[0] for b in bars], self.cov_types)

        plt.tight_layout()
        # plt.show()
        return(plt)

    def predict(self, X):
        clf = self.get_best_model()
        y_pred = clf.predict(X)
        return(y_pred)

    def plot_best_model(self, feat_x, feat_y):
        plt.figure(figsize=(12, 6))
        splot = plt.subplot(1, 1, 1)

        color_iter = itertools.cycle(['k', 'r', 'b', 'g', 'c', 'm', 'y'])
        clf = self.get_best_model()

        for i, (mean, covar, color) in enumerate(zip(clf.means_,
                                                     clf.covariances_,
                                                     color_iter)):
            if len(covar.shape) < 2:
                tmp = np.zeros((2, 2))
                np.fill_diagonal(tmp, covar)
                covar = tmp
            elif covar.shape[0] != covar.shape[1]:
                covar = np.diag(covar)

            v, w = linalg.eigh(covar)
            if not np.any(self.y_pred == i):
                continue

            plt.scatter(self.X[self.y_pred == i][feat_x],
                        self.X[self.y_pred == i][feat_y],
                        5, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180 * angle / np.pi  # convert to degrees
            v *= 4
            ell = mpl.patches.Ellipse(mean, v[0], v[1],
                                      180 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(.5)
            splot.add_artist(ell)

        plt.title('Selected GMM')
        plt.show()

    def clusters_vs_true_labels(self):
        self.y_pred = self.predict(self.X)
        num_true_classes = len(set(self.y))
        print(mt.confusion_matrix(self.y, self.y_pred)[0:num_true_classes, :])
