"""
Preprocessing package for KBO detection in JWST data

This package contains utilities for loading, calibrating,
and aligning JWST FITS data for KBO detection.
"""

from .fits_loader import load_fits_file
from .calibration import subtract_background, clean_image
from .alignment import align_images

__all__ = [
    'load_fits_file',
    'subtract_background',
    'clean_image',
    'align_images'
]