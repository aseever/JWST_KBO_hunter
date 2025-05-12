"""
mast - Package for JWST KBO detection pipeline

This package provides tools to search MAST for JWST observations,
filter for KBO candidates, and download FITS files for processing.
"""

# Import key modules
from . import utils
from . import search
from . import filter
from . import download

# Export key constants
from .utils import KBO_DETECTION_CONSTANTS

# Version information
__version__ = '0.1.0'