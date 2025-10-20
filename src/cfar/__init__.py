"""CFAR algorithms and tracking utilities."""

from .cfar1d import ca_cfar_1d
from .cfar2d import ca_cfar_2d
from .tracking import PeakTracker

__all__ = ["ca_cfar_1d", "ca_cfar_2d", "PeakTracker"]

