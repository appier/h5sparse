import pkg_resources
from .h5sparse import Group, File, Dataset  # noqa: F401

__version__ = pkg_resources.get_distribution("h5sparse").version
