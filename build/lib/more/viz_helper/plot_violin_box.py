import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_violin_box(data,
                    x,
                    y,
                    bw=1,
                    cut=0,
                    scale='area',
                    figsize=(10, 5),
                    xrot=0,
                    yrot=0,
                    xlab=float('nan'),
                    ylab=float('nan')):
    """
        A functions to visualize a continuous field vs. a categorical field)
    data: Dataframe to plot
    x: Column name to plot on the X-axis (categorical variable)
    y: Column name to plot on the Y-axis (continuous variable)
    bw: (Default = 1) Bandwidth for the Violon plot
    cut = (Default = 0) Cut option for the Violin plot
    scale: (Default = Area) Scale option for the Violin plot
    figsize: Default = (10,5). Overall figure size for the Violin and Box plots
    xrot: (Default = 0). Degrees to rotate the X label by
    yrot: (Default = 0). Degrees to rotate the Y label by
    xlab: Label to use for the X-axis
    ylab: Label to use for the Y-axis
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.xticks(rotation=xrot)

    # Violin Plot
    ax = sns.violinplot(x=x, y=y, data=data, palette="RdBu",
                        cut=0, ax=axes[0], scale=scale,
                        inner='quartile', bw=bw)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot)  # , fontsize = 8
    ax.set_yticklabels(ax.get_yticklabels(), rotation=yrot)  # , fontsize = 8
    if (pd.notna(xlab)):
        ax.set_xlabel(xlab)
    if (pd.notna(ylab)):
        ax.set_ylabel(ylab)

    # Box Plot
    ax = sns.boxplot(x=x, y=y, data=data, palette="RdBu", ax=axes[1])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot)  # , fontsize = 8
    ax.set_yticklabels(ax.get_yticklabels(), rotation=yrot)  # , fontsize = 8
    if (pd.notna(xlab)):
        ax.set_xlabel(xlab)
    if (pd.notna(ylab)):
        ax.set_ylabel(ylab)
