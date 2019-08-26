import matplotlib.pyplot as plt
import seaborn as sns


def plot_nulls(data, figsize=None):
    """
        data: Dataframe to plot
    """

    # Run heatmap again to validate nulls are handled
    plt.figure(figsize=figsize)
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
