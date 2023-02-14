"""The mag scan after interpolation."""

import numpy as np

from mag_utils.mag_utils.visualization.plotter import plot_interpolated_scan
from mag_utils.mag_utils.saver import save_as_h5
from mag_utils.mag_utils.saver.interpolated import save_as_json, save_as_tiff


class InterpolatedScan:
    """object containing the data of scan after interpolation."""

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 b: np.ndarray,
                 mask: np.ndarray,
                 interpolation_method: str):
        """
        Create an instance of InterpolatedScan object.

        Args:
            x: matrix of the interpolated scan coordinate [utm].
            y: matrix of the interpolated scan coordinate [utm].
            b: matrix of the interpolated magnetic field [nT].
            mask: binary matrix that tells where the values are interpolated vs extrapolated.
            interpolation_method: name of the interpolation method applied.
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.b = np.asarray(b)
        self.mask = np.asarray(mask)
        self.interpolation_method = interpolation_method

    def __eq__(self, other):
        """
        Check whether the two objects are equals.

        Args:
            other: InterpolatedScan object.

        Returns:
            boolean that indicate whether the objects are the same.
        """
        if self.__class__ == other.__class__:
            is_equals = (self.x == other.x).all() and \
                        (self.y == other.y).all() and \
                        (self.b == other.b).all() and \
                        (self.mask == other.mask).all() and \
                        (self.interpolation_method == other.interpolation_method)
        else:
            raise TypeError(f"Can't compare object of type {other.__class__} to {self.__class__}")

        return is_equals

    def plot(self, levels: int = 50, show_extrapolated: bool = False, ax=None, **kwargs):
        """
        Plot the masked data using matplotlib contourf function.

        Giving an ax will give you the option to keep editing the plot as you like.

        Examples:
            import matplotlib.pyplot as plt
            from mag_utils.loader import blackwidow

            # Showing the plot.
            widow_path = "some/path/scan.txt"
            scan = blackwidow.load(widow_path)
            scan.interpolate('Kriging', dist_between_points=0.5, inplace=True)
            scan.interpolated_data.plot()

            # plotting on a custom axis

            fig, ax = plt.subplots()
            plot_data = scan.interpolated_data.plot(ax=ax)
            fig.colorbar(plot_data)
            ax.set_title('I love myfat')
            plt.show()

        Args:
            levels: The number of contour lines.
            ax: matplotlib axs. If given will plot the graph on it, without showing.
                Useful for creating subplots.
            show_extrapolated: whether or not to plot the extrapolated data.
            **kwargs: matplotlib contourf kwargs.

        Returns:
            None in no ax is given. The output of the contourf plot if ax is given.
        """
        masked_b = self.b.copy()
        if not show_extrapolated:
            masked_b[~self.mask] = None

        return plot_interpolated_scan(self.x, self.y, masked_b, levels=levels, ax=ax, **kwargs)

    def save(self, path: str):
        """
        Save the object data in a file according to the file type.

        Supported file types: h5, tif, json.

        Args:
            path: The output file path.
        """
        if path.lower().endswith(".h5") or path.lower().endswith(".hdf5"):
            save_as_h5.save(output_path=path, scan=self)
        elif path.lower().endswith(".tif"):
            save_as_tiff(output_path=path,
                         x=self.x,
                         y=self.y,
                         b=self.b,
                         mask=self.mask,
                         interpolation_method=self.interpolation_method)
        elif path.lower().endswith(".json"):
            save_as_json(output_path=path,
                         x=self.x,
                         y=self.y,
                         b=self.b,
                         mask=self.mask,
                         interpolation_method=self.interpolation_method)
