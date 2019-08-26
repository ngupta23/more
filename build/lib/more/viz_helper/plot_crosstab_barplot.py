import pandas as pd
import matplotlib.pyplot as plt


def plot_crosstab_barplot(data,
                          y=None,
                          by=None,
                          figsize=(12, 6),
                          ylab=float('nan'),
                          title=None,
                          loc='best',
                          anchor=(1, 1)):
    """
    data: Dataframe to plot
    y: Column name to crosstabulate by
        by: Column name for which to plot the percentage breakdown
        figsize: (10,6). Overall figure size for the Violin and Box plots
    ylab: Label to use for the Y-axis
    title: (Default = None) Title of the legend
    loc: (Default = 'best') Legend location
    anchor: Default = (1,1). Specfies where to anchor the legend
    """
    crosstab_data = pd.crosstab(data[by], data[y].astype(bool))
    crosstab_rate = round(crosstab_data.div(
        crosstab_data.sum(axis=1), axis=0), 2)

    ax = crosstab_rate.plot(kind='barh', stacked=True, figsize=figsize
                            # ,colormap="PiYG"
                            )
    if (pd.notna(ylab)):
        ax.set_ylabel(ylab)

    plt.legend(title=title, loc=loc, fancybox=True, bbox_to_anchor=anchor)
