"""Wrapper functions for matplotlib specific for mag data."""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable


UP_DISPLAY_OFFSET = 100


def plot_horizontal_scan(x, y, c=None, ax: Axes = None, **kwargs):
    """
    Plots a scatter plot of the x,y points.

    Args:
        x: x coordinates, shape [N]
        y: y coordinates, shape [N]
        c: color values for each (x,y) point, shape [N]
        ax: matplotlib axs. If given will plot the graph on it, without showing the figure.
            Useful for creating subplots.
        **kwargs: plot and matplotlib scatter kwargs.
                use center=True to center the plot at (0,0)
                use colorbar=True to display colorbar
    Returns:
        If no ax is given in input: None.
        If ax is given in input: output of the scatter plot.


    """
    show_colorbar = True

    if 'center' in kwargs.keys():
        if kwargs['center']:
            # center x and y
            x = (x - np.min(x)) - (np.max(x) - np.min(x)) / 2
            y = (y - np.min(y)) - (np.max(y) - np.min(y)) / 2
        del kwargs['center']

    if 'colorbar' in kwargs.keys():
        show_colorbar = kwargs['colorbar']
        del kwargs['colorbar']

    if ax is None:
        plt.scatter(x, y, c=c, cmap='jet', **kwargs)

        plt.xlabel('x utm [m]')
        plt.ylabel('y utm [m]')
        plt.gca().axis('equal')

        if show_colorbar:
            plt.colorbar()
        plt.show()
        return None

    ax.set_xlabel('x utm [m]')
    ax.set_ylabel('y utm [m]')
    ax.axis('equal')

    scatter_out = ax.scatter(x, y, c=c, cmap='jet', **kwargs)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(scatter_out, cax=cax, orientation='vertical')
    return scatter_out


def plot_interpolated_scan(x, y, c, levels: int = 50, ax: Axes = None, **kwargs):
    """
        Plots a scatter plot of the x,y points.

        Args:
            x: x coordinates, shape [N, M]
            y: y coordinates, shape [N, M]
            c: color values of the data, shape [N, M]
            levels:  the number of contour lines.
            ax: matplotlib axs. If given will plot the graph on it, without showing.
                Useful for creating subplots.
            **kwargs: plot and matplotlib contourf kwargs.
                use center=True to center the plot at (0,0)
                use colorbar=True to display colorbar
        Returns:
            If no ax is given in input: None.
            If ax is given in input: output of the contourf plot.


        """
    show_colorbar = False

    if 'center' in kwargs.keys():
        if kwargs['center']:
            # center x and y
            x = (x - np.min(x)) - (np.max(x) - np.min(x)) / 2
            y = (y - np.min(y)) - (np.max(y) - np.min(y)) / 2
        del kwargs['center']

    if 'colorbar' in kwargs.keys():
        show_colorbar = kwargs['colorbar']
        del kwargs['colorbar']

    if ax is None:
        plt.contour(x, y, c, colors='black', levels=levels)
        plt.contourf(x, y, c, cmap='jet', levels=50, **kwargs)

        plt.xlabel('x utm [m]')
        plt.ylabel('y utm [m]')
        plt.gca().axis('equal')

        if show_colorbar:
            plt.colorbar()
        plt.show()
        return None

    ax.contour(x, y, c, colors='black', levels=levels)
    ax.set_xlabel('x utm [m]')
    ax.set_ylabel('y utm [m]')
    ax.axis('equal')
    cont_output = ax.contourf(x, y, c, cmap="jet", levels=levels, **kwargs)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cont_output, cax=cax, orientation='vertical')

    return cont_output


def plot_pso_output(x, y, target_x, target_y, ax: Axes = None, **kwargs):

    show_colorbar = True

    if ax is None:
        plt.scatter(x, y, c=None, cmap='jet', **kwargs)
        plt.scatter([[target_x]], [[target_y]], c="black")

        plt.xlabel('x utm [m]')
        plt.ylabel('y utm [m]')
        plt.gca().axis('equal')

        if show_colorbar:
            plt.colorbar()
        plt.show()
        return None

    ax.set_xlabel('x utm [m]')
    ax.set_ylabel('y utm [m]')
    ax.axis('equal')

    scatter_out = ax.scatter(x, y, c=c, cmap='jet', **kwargs)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(scatter_out, cax=cax, orientation='vertical')
    return scatter_out
