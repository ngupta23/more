import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math


def plot_scatter_one_vs_all(data,
                            y=None,
                            hue=None,
                            rows=None,
                            cols=None,
                            figsize=None,
                            xrot=0,
                            yrot=0,
                            alpha=0.7):
    """
    data: (Mandatory) Dataframe to plot
    y: (Mandatory) Variable to plot on the y-axis
    hue: Used to define the column in data that is used to color code the
        scatterplot
    rows: (Autocalculated by default) Specifies number of rows in the plots
    cols: (Autocalculated by default)Specifies number of columns in the plots
    figsize: (Autocalculated by default) Overall figure size for the plot
             matrix. Autocalculated by default.
    xrot: (Default = 0) Degrees to rotate the X label by
    yrot: (Default = 0) Degrees to rotate the Y label by
    alpha: (Default = 0.7) Transparency to use for the scatterplots
    """

    """
    TODO:
        (1) Check for datatypes to make sure they are correct
        (2) Values not showing up on axis
    """

    if (y is None):
        raise Exception("Please specify the variable to plot on the Y-axis")

    num_columns = len(data.columns)

    if (hue is None):
        num_scatter = num_columns - 1
    elif (hue == y):  # Both Hue and Y are Same
        num_scatter = num_columns - 1
    else:
        num_scatter = num_columns - 2

    # Both are None, then code decides
    if ((rows is None) and (cols is None)):
        if (num_scatter % 5 == 0):
            cols = 5
        elif (num_scatter % 4 == 0):
            cols = 4
        elif (num_scatter % 3 == 0):
            cols = 3
        elif (num_scatter % 2 == 0):
            cols = 2
        else:
            cols = 1
        rows = math.ceil(num_scatter/cols)
    # If number of rows is none, then number of columns gets decided by code
    elif (rows is None):
        rows = math.ceil(num_scatter/cols)
    # If number of columns is none, then number of rows gets decided by code
    elif (cols is None):
        cols = math.ceil(num_scatter/rows)
    # When both are specified by user
    elif (rows * cols < num_scatter):
        warnings.warn(
            "Number of rows and columns specified is less than number of "
            "scatter plots to be plotted. Some scatterplots will be omitted")

    num_plots_requested = rows * cols

    if (figsize is None):
        if (cols > 2):
            loWidth = 4 * cols
            loHeight = 4 * rows
        else:
            loWidth = 8 * cols
            loHeight = 8 * rows
        figsize = (loWidth, loHeight)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

    cols_to_plot = data.columns

    # Remove columns that you dont want to plot
    cols_to_plot = data.columns.drop(y)
    if (hue is not None):
        cols_to_plot = data.columns.drop([y, hue])

    for i, column in enumerate(cols_to_plot):
        if i < num_plots_requested:
            if (rows == 1 and cols == 1):
                axis = axes
            elif ((rows == 1 and cols != 1) or (cols == 1 and rows != 1)):
                axis = axes[i]
            else:
                axis = axes[i//cols, i % (cols)]

            ax = sns.scatterplot(data=data, y=y, x=column,
                                 hue=hue, alpha=alpha, ax=axis)
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=xrot)  # , fontsize = 8
            ax.set_yticklabels(ax.get_yticklabels(),
                               rotation=yrot)  # , fontsize = 8

    plt.tight_layout()
