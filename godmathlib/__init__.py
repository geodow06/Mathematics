print(f'Invoking __init__.py for {__name__}')

from . import linear_algebra, calculus
from .linear_algebra import *
from .calculus import *
from .constants import *
from .utils import *

__all__ = [
        'linear_algebra',
        'calculus',
        'constants',
        'utils'
        ]

__all__.extend(linear_algebra.__all__)
__all__.extend(constants.__all__)
__all__.extend(utils.__all__)
__all__.extend(calculus.__all__)