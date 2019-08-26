import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def perform_pca(data, n_components=3, y=None):
    """
    data = Data on which to perform PCA
    n_components = 3. This is the number of Principal Components to
                      consider for the analysis
    """
    # Separating and standardizing the features
    features = data.copy(deep=True)
    if (y is not None):
        features = data.drop(y, axis=1)
    x = StandardScaler().fit_transform(features)

    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(x)
    principal_comp = pd.DataFrame(
            data=principalComponents,
            columns=['Principal Component '
                     + str(i) for i in range(1, (n_components+1))])

    if (y is not None):
        final_df = pd.concat([principal_comp, data[y]], axis=1)
    else:
        final_df = principal_comp.copy(deep=True)
    print("% variance explained by each Principal Component: " +
          str(np.round(pca.explained_variance_ratio_ * 100, 2)))
    print("% Total variance explained by all Principal Components: " +
          str(round(sum(pca.explained_variance_ratio_) * 100, 2)) + "%")
    return (final_df)


def plot_prin_comps(data,
                    figsize=None,
                    hue=None,
                    alpha=0.7,
                    bins=None,
                    kde=True,
                    xrot=0,
                    yrot=0):
    """
    data = Data returned by the perform_pca function
    figsize = Overall figure size for the overall plot.
              Autocalculated by default based on number of columns
    hue = 'promotion_max'. Column name to use to color code the scatter plot
    alpha = 0.7. Transparency to use for the scatterplots
    bins = 50. Number of bins to use in the histogram on the diaginals
    kde = True. Should KDE be plotted on the diagonals
    xrot = 0
    yrot = 0
    """
    # Define Shape and size of matrix
    arSquare = 1
    if (hue is not None):
        # remove the column hue for size calculations
        arSquare = data.shape[1]-1
    else:
        arSquare = data.shape[1]

    if (figsize is None):
        figsize = (4*arSquare, 4*arSquare)

    fig, axes = plt.subplots(nrows=arSquare, ncols=arSquare, figsize=figsize)

    cols_to_plot = data.columns
    if (hue is not None):
        cols_to_plot = data.columns.drop(hue)

    for i, column1 in enumerate(cols_to_plot):
        for j, column2 in enumerate(cols_to_plot):
            if (i == j):
                # draw histogram and KDE on diagonal
                ax = sns.distplot(
                    data[data[column1].notna()][column1],
                    kde=kde, bins=bins, ax=axes[i, j])
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=xrot)  # , fontsize = 8
                ax.set_yticklabels(ax.get_yticklabels(),
                                   rotation=yrot)  # , fontsize = 8

            if (i < j):
                # only draw scatterplot on one side of the diagnal
                ax = sns.scatterplot(data=data, y=column1,
                                     x=column2, hue=hue, ax=axes[i, j])
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=xrot)  # , fontsize = 8
                ax.set_yticklabels(ax.get_yticklabels(),
                                   rotation=yrot)  # , fontsize = 8

    plt.tight_layout()
