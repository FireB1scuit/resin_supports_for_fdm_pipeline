"""Auto-orient any STL and give it resin-style supports that survive on FDM."""

from .types import Orientation, SupportBuild, SupportParams, SupportPoint

__version__ = "0.1.0"

__all__ = ["SupportParams", "SupportPoint", "Orientation", "SupportBuild", "__version__"]
