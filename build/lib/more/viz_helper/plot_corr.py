import matplotlib.pyplot as plt
import seaborn as sns


def plot_corr(data, figsize=None, xrot=0, yrot=0):
    """
    corr_mat: Correlation matrix to visualize
    figsize: Overall figure size for the overall plot
    xrot: (Default = 0) Rotate x-labels
    yrot: (Default = 0) Rotate y-labels
    """
    fig, axis = plt.subplots(1, 1, figsize=figsize)
    corr_mat = data.corr()
    sns.heatmap(round(corr_mat, 2), annot=True, cmap='RdBu', vmin=-1, vmax=1)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=xrot)
    axis.set_yticklabels(axis.get_yticklabels(), rotation=yrot)
