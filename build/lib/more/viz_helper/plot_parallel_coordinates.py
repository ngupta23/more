import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates


def plot_parallel_coordinates(data,
                              by,
                              sample=True,
                              frac=1.0,
                              normalize=True,
                              figsize=(8, 6),
                              legend_title=None,
                              legend_loc='best',
                              legend_anchor=(1, 1),
                              xrot=0,
                              random_state=1):
    """
    data: Dataframe to use for plotting the parallel coordinates plot
    by: Column name to categorize the plot by
        Currently only supports a column with 2 classes having values 0 and 1
    sample: (Default = True) Should the data be sampled
            (May be necessary for large datasets)
    frac: (Default = 1.0 --> Take full dataset)
          If sample = True, then use this sampling fraction
    normalize: (Defauult = True) Should the data be normalized.
                Make True if columns are not on same scale already
    """

    if(sample):
        df_sub = data.sample(frac=frac, replace=False,
                             random_state=random_state)
    else:
        df_sub = data.copy(deep=False)

    # This plot is more meaningful when values are normalized
    # They are on the same scale here so no need to normalize
    if (normalize):
        df_normalized = (df_sub-df_sub.mean())/(df_sub.std())
        df_normalized[by] = df_sub[by]
    else:
        df_normalized = df_sub.copy(deep=False)

    plt.figure(figsize=figsize)

    ax = parallel_coordinates(df_normalized, by, colormap='viridis')
    if (legend_title is None):
        legend_title = by
    plt.legend(title=legend_title, loc=legend_loc,
               fancybox=True, bbox_to_anchor=legend_anchor)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot)  # , fontsize = 8
