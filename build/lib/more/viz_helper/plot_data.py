import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math


def plot_data(data,
              kind='dist',
              rows=None,
              cols=None,
              kde=False,
              bins=None,
              figsize=None,
              xrot=0,
              yrot=0):
    """
    data = Dataframe to plot
    kind = 'dist'.
           Specifies kind of plot, Options are 'dist' (default), 'count'
    rows = 3. Specifies number of rows in the plots
    rows = 3. Specifies number of columns in the plots
    kde = False. If kind = 'dist', should the KDE be plotted?
    bins = If kind = 'dist', this argument specifies the number of bins
           that should be plotted
    figsize = Overall figure size for the plot matrix.
              Autocalculated by default.
    xrot = 0. Degrees to rotate the X label by
    yrot = 0. Degrees to rotate the Y label by
    """

    """
    TODO: Add checks for variable types for each type of plot
    """

    if (kind == 'dist'):
        cat_columns = data.select_dtypes(
            include=['object', 'category']).columns
        data = data.select_dtypes(include=['number'])
        if (len(cat_columns) > 0):
            warnings.warn(
                "The data has categorical columns {} for which a distribution "
                "will not be plotted".format(cat_columns))
    elif (kind == 'count'):
        num_columns = data.select_dtypes(include=['number']).columns
        data = data.select_dtypes(include=['object', 'category'])
        if (len(num_columns) > 0):
            warnings.warn(
                "The data has numeric columns {} for which a count plot will "
                "not be plotted".format(num_columns))
    else:
        raise Exception("kind = " + str(kind) + " is not supported yet")

    num_columns = len(data.columns)

    # Both are None, then code decides
    if ((rows is None) and (cols is None)):
        if (num_columns % 5 == 0):
            cols = 5
        elif (num_columns % 4 == 0):
            cols = 4
        elif (num_columns % 3 == 0):
            cols = 3
        elif (num_columns % 2 == 0):
            cols = 2
        else:
            cols = 1
        rows = math.ceil(num_columns/cols)
    # If number of rows is none, then number of columns gets decided by code
    elif (rows is None):
        rows = math.ceil(num_columns/cols)
    # If number of columns is none, then number of rows gets decided by code
    elif (cols is None):
        cols = math.ceil(num_columns/rows)
    # When both are specified by user
    elif (rows * cols < num_columns):
        warnings.warn(
            "Number of rows and columns specified is less than number of "
            "scatter plots to be plotted. Some scatterplots will be omitted")

    num_plots_requested = rows * cols

    if (figsize is None):
        if (cols > 2):
            loWidth = 4 * cols
            loHeight = 4 * rows
        else:
            loWidth = 4 * cols
            loHeight = 4 * rows
        figsize = (loWidth, loHeight)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

    cols_to_plot = data.columns

    for i, column in enumerate(cols_to_plot):
        if i < num_plots_requested:
            if (rows == 1 and cols == 1):
                axis = axes
            elif ((rows == 1 and cols != 1) or (cols == 1 and rows != 1)):
                axis = axes[i]
            else:
                axis = axes[i//cols, i % (cols)]

            if (kind == 'dist'):
                ax = sns.distplot(data[data[column].notna()]
                                  [column], kde=kde, bins=bins, ax=axis)
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=xrot)  # , fontsize = 8
                ax.set_yticklabels(ax.get_yticklabels(),
                                   rotation=yrot)  # , fontsize = 8
            elif (kind == 'count'):
                ax = sns.countplot(x=column, data=data, ax=axis)
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=xrot)  # , fontsize = 8
                ax.set_yticklabels(ax.get_yticklabels(),
                                   rotation=yrot)  # , fontsize = 8
            else:
                pass

    plt.tight_layout()
