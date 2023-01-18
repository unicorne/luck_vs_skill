import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_points_as_histogramm(points, title, figsize=(12, 6), stepsize=2):
    """
    Plot the points of a league as a histogramm.

    Parameters
    ----------
    points : numpy.ndarray
        The points of the league.
    title : str
        The title of the plot.
    figsize : tuple, optional
        The size of the figure. The default is (12,6).
    stepsize : int, optional
        The stepsize of the x-axis. The default is 2. Determines the number of bins.
    """

    fig = plt.figure(figsize=figsize)

    x = np.arange(0, max(points) + 1, stepsize)

    # the histogram of the data
    n, bins, patches = plt.hist(
        points, len(x), density=False, facecolor="g", alpha=0.75
    )
    plt.xlabel("Points")
    plt.ylabel("#Teams")
    plt.title(title)
    plt.show()
