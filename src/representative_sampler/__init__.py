# read version from installed package
from importlib.metadata import version
import logging
__version__ = version("representative_sampler")
__package_name__ = __name__
