"""Registry of all interpolation methods."""
from .kriging import Kriging
from .linear import Linear
from .rbf import RBF
from .nearest import Nearest
from .old_idw import OldIdw
from .old_v21 import OldV21
from .cubic_spline import CubicSpline

interpolation_registry = {Nearest.__name__: Nearest,
                          RBF.__name__: RBF,
                          Linear.__name__: Linear,
                          Kriging.__name__: Kriging,
                          OldIdw.__name__: OldIdw,
                          OldV21.__name__: OldV21,
                          CubicSpline.__name__: CubicSpline}
