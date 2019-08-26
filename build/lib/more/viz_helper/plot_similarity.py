import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import distance


def plot_similarity(data,
                    y,
                    sample=True,
                    frac=1.0,
                    random_state=1,
                    figsize=None):
    """
    data: Dataframe to plot
    y: Column to sort the data by before computing similarity
       (usually the output)
    sample: (Default = True) Should the data be sampled.
            Advisable for larger darasets otherewise it takes a long time
    frac = (Default = 1.0 --> Take full dataset)
            What fraction of the data should be sampled
    """
    sns.set(rc={'image.cmap': 'cubehelix'})
    # get a subset of the data, and normalize it
    if (sample is True):
        df_sub = data.sample(frac=frac, replace=False,
                             random_state=random_state)
    df_normalized = (df_sub-df_sub.mean())/(df_sub.std())
    df_sorted = df_normalized.sort_values(by=y)
    Y = distance.pdist(df_sorted, "euclidean")
    A = distance.squareform(Y)
    S = 0.5/(1+np.exp(A))  # convert from distance to similarity
    # plot the similarity matrix using seaborn color utilities

    plt.figure(figsize=figsize)
    plt.pcolormesh(S)
    plt.colorbar()
