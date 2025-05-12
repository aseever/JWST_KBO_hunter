"""
coordinates.py - Coordinate conversion utilities for KBO catalog lookup

This module provides utilities for working with different coordinate systems,
conversion between coordinate representations, and handling of KBO-specific
coordinate calculations.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union
from astropy.coordinates import SkyCoord, ICRS, Galactic, AltAz, EarthLocation
from astropy.coordinates import get_body, get_sun
from astropy.time import Time
import astropy.units as u
from astropy.wcs import WCS

def degrees_to_hms(ra: float, dec: float) -> Tuple[str, str]:
    """
    Convert decimal degrees to HMS/DMS representation.
    
    Args:
        ra: Right ascension in decimal degrees.
        dec: Declination in decimal degrees.
        
    Returns:
        Tuple of (RA in HH:MM:SS format, Dec in DD:MM:SS format)
    """
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    ra_hms = coord.ra.to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
    dec_dms = coord.dec.to_string(unit=u.degree, sep=':', precision=2, pad=True, alwayssign=True)
    return ra_hms, dec_dms

def hms_to_degrees(ra_hms: str, dec_dms: str) -> Tuple[float, float]:
    """
    Convert HMS/DMS representation to decimal degrees.
    
    Args:
        ra_hms: Right ascension in HH:MM:SS format.
        dec_dms: Declination in DD:MM:SS format.
        
    Returns:
        Tuple of (RA in decimal degrees, Dec in decimal degrees)
    """
    coord = SkyCoord(ra_hms, dec_dms, unit=(u.hourangle, u.deg), frame='icrs')
    return coord.ra.degree, coord.dec.degree

def pixel_to_radec(x: float, y: float, wcs: WCS) -> Tuple[float, float]:
    """
    Convert pixel coordinates to RA/Dec using WCS information.
    
    Args:
        x: X pixel coordinate
        y: Y pixel coordinate
        wcs: WCS object containing the coordinate transformation
        
    Returns:
        Tuple of (RA, Dec) in degrees
    """
    if wcs is None:
        raise ValueError("WCS information is required for pixel to RA/Dec conversion")
    
    # Convert pixel to world coordinates
    # Adding 1 to account for FITS (1-based) vs Python (0-based) indexing
    sky = wcs.pixel_to_world(x, y)
    
    # Extract RA and Dec in degrees
    ra = sky.ra.degree
    dec = sky.dec.degree
    
    return ra, dec

def radec_to_pixel(ra: float, dec: float, wcs: WCS) -> Tuple[float, float]:
    """
    Convert RA/Dec coordinates to pixel coordinates using WCS information.
    
    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        wcs: WCS object containing the coordinate transformation
        
    Returns:
        Tuple of (X, Y) pixel coordinates
    """
    if wcs is None:
        raise ValueError("WCS information is required for RA/Dec to pixel conversion")
    
    # Create SkyCoord object
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    
    # Convert to pixel coordinates
    x, y = wcs.world_to_pixel(coords)
    
    return x, y

def precess_coordinates(ra: float, dec: float, from_epoch: str, to_epoch: str) -> Tuple[float, float]:
    """
    Precess coordinates from one epoch to another.
    
    Args:
        ra: Right ascension in decimal degrees.
        dec: Declination in decimal degrees.
        from_epoch: Source epoch (e.g., 'J2000').
        to_epoch: Target epoch (e.g., 'J2022').
        
    Returns:
        Tuple of (RA in decimal degrees, Dec in decimal degrees) in the new epoch.
    """
    # Create SkyCoord at the source epoch
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs',
                    obstime=Time(from_epoch))
    
    # Transform to the target epoch
    coord_new = coord.transform_to(ICRS(obstime=Time(to_epoch)))
    
    return coord_new.ra.degree, coord_new.dec.degree

def equatorial_to_ecliptic(ra: float, dec: float) -> Tuple[float, float]:
    """
    Convert equatorial coordinates to ecliptic coordinates.
    
    Args:
        ra: Right ascension in decimal degrees.
        dec: Declination in decimal degrees.
        
    Returns:
        Tuple of (ecliptic longitude in degrees, ecliptic latitude in degrees)
    """
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    ecliptic = coord.transform_to('geocentrictrueecliptic')
    return ecliptic.lon.degree, ecliptic.lat.degree

def ecliptic_to_equatorial(lon: float, lat: float) -> Tuple[float, float]:
    """
    Convert ecliptic coordinates to equatorial coordinates.
    
    Args:
        lon: Ecliptic longitude in decimal degrees.
        lat: Ecliptic latitude in decimal degrees.
        
    Returns:
        Tuple of (RA in decimal degrees, Dec in decimal degrees)
    """
    coord = SkyCoord(lon=lon*u.degree, lat=lat*u.degree, frame='geocentrictrueecliptic')
    equatorial = coord.transform_to('icrs')
    return equatorial.ra.degree, equatorial.dec.degree

def calculate_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """
    Calculate angular separation between two celestial positions.
    
    Args:
        ra1: Right ascension of first position in decimal degrees.
        dec1: Declination of first position in decimal degrees.
        ra2: Right ascension of second position in decimal degrees.
        dec2: Declination of second position in decimal degrees.
        
    Returns:
        Angular separation in decimal degrees.
    """
    # Use SkyCoord objects instead of low-level functions
    coord1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    coord2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
    
    # Calculate separation
    return coord1.separation(coord2).degree

def coordinates_to_altaz(ra: float, dec: float, time: Time, 
                       location: EarthLocation) -> Tuple[float, float]:
    """
    Convert equatorial coordinates to horizontal (alt/az) coordinates.
    
    Args:
        ra: Right ascension in decimal degrees.
        dec: Declination in decimal degrees.
        time: Observation time as astropy Time object.
        location: Observer location as astropy EarthLocation object.
        
    Returns:
        Tuple of (azimuth in degrees, altitude in degrees)
    """
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    altaz = coord.transform_to(AltAz(obstime=time, location=location))
    return altaz.az.degree, altaz.alt.degree

def format_target_for_skybot(ra: float, dec: float, epoch: Time) -> Dict[str, Any]:
    """
    Format target coordinates for SkyBoT query.
    
    Args:
        ra: Right ascension in decimal degrees.
        dec: Declination in decimal degrees.
        epoch: Observation time as astropy Time object.
        
    Returns:
        Dictionary with formatted coordinates for SkyBoT.
    """
    return {
        "RA": ra,
        "DEC": dec,
        "EPOCH": epoch.isot
    }

def format_target_for_mpc(ra: float, dec: float, epoch: Time) -> Dict[str, Any]:
    """
    Format target coordinates for MPC query.
    
    Args:
        ra: Right ascension in decimal degrees.
        dec: Declination in decimal degrees.
        epoch: Observation time as astropy Time object.
        
    Returns:
        Dictionary with formatted coordinates for MPC.
    """
    ra_hms, dec_dms = degrees_to_hms(ra, dec)
    return {
        "ra": ra_hms,
        "dec": dec_dms,
        "date": epoch.iso.split()[0]  # Just the date part
    }

def estimate_rate_of_motion(distance_au: float) -> float:
    """
    Estimate typical rate of motion for a KBO at a given distance.
    
    Args:
        distance_au: Distance in AU.
        
    Returns:
        Typical motion rate in arcseconds per hour.
    """
    # Approximate formula based on typical KBO motion
    # Motion scales roughly as distance^(-3/2)
    # Normalized to ~3 arcsec/hour at 40 AU
    return 3.0 * (40.0 / distance_au)**(3/2)

def is_near_ecliptic(dec: float, max_deviation_deg: float = 5.0) -> bool:
    """
    Check if coordinates are near the ecliptic plane.
    
    Args:
        dec: Declination in decimal degrees.
        max_deviation_deg: Maximum allowed deviation from ecliptic in degrees.
        
    Returns:
        True if coordinates are near the ecliptic plane.
    """
    # Get ecliptic coordinates for the current epoch
    _, ecliptic_lat = equatorial_to_ecliptic(0.0, dec)
    
    # Check if close to ecliptic plane
    return abs(ecliptic_lat) <= max_deviation_deg