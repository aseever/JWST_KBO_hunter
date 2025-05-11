"""
Utility Functions for Catalog Lookup

This module provides utility functions for the catalog lookup system,
including coordinate conversions, caching, and API rate limiting.

Main components:
- Coordinates: Functions for coordinate transformations and calculations
- Cache: Query result caching to improve performance and reduce API calls
- RateLimiter: API rate limiting to respect service usage policies
"""

# Import utility modules
from catalog_lookup.utils.coordinates import (
    pixel_to_radec, 
    radec_to_pixel,
    calculate_separation,
    convert_equatorial_to_ecliptic,
    calculate_motion_vector
)

from catalog_lookup.utils.cache import (
    CacheManager,
    get_cached_result,
    cache_result,
    clear_cache
)

from catalog_lookup.utils.rate_limiter import (
    RateLimiter,
    get_rate_limiter
)

# Default timeouts for API calls (in seconds)
DEFAULT_TIMEOUTS = {
    'connect': 10.0,  # Connection timeout
    'read': 30.0      # Read timeout
}

# Default cache settings
DEFAULT_CACHE_EXPIRY = 86400  # 24 hours in seconds
DEFAULT_CACHE_SIZE = 1000     # Maximum number of items in cache

# Coordinate system constants
COORDINATE_SYSTEMS = {
    'icrs': 'ICRS (International Celestial Reference System)',
    'j2000': 'J2000 (Equinox J2000)',
    'ecliptic': 'Ecliptic',
    'galactic': 'Galactic'
}

__all__ = [
    'pixel_to_radec',
    'radec_to_pixel',
    'calculate_separation',
    'convert_equatorial_to_ecliptic',
    'calculate_motion_vector',
    'CacheManager',
    'get_cached_result',
    'cache_result',
    'clear_cache',
    'RateLimiter',
    'get_rate_limiter',
    'DEFAULT_TIMEOUTS',
    'DEFAULT_CACHE_EXPIRY',
    'DEFAULT_CACHE_SIZE',
    'COORDINATE_SYSTEMS'
]