"""
Utils package for KBO catalog lookup

This package contains utility modules for coordinate transformations,
caching, and rate limiting.
"""

from catalog_lookup.utils.coordinates import (
    degrees_to_hms,
    hms_to_degrees,
    pixel_to_radec,
    radec_to_pixel,
    precess_coordinates,
    equatorial_to_ecliptic,
    ecliptic_to_equatorial,
    calculate_separation,
    coordinates_to_altaz,
    format_target_for_skybot,
    format_target_for_mpc,
    estimate_rate_of_motion,
    is_near_ecliptic
)

from catalog_lookup.utils.cache import QueryCache, cached, default_cache
from catalog_lookup.utils.rate_limiter import (
    RateLimiter, 
    WindowRateLimiter,
    mpc_rate_limiter,
    jpl_rate_limiter,
    skybot_rate_limiter,
    panstarrs_rate_limiter
)