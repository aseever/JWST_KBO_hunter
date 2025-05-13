"""
mast/filter/core.py - Core filtering utilities for JWST KBO detection

This module provides the fundamental filtering operations for JWST observations
to identify promising KBO candidates, focusing on instrument selection, wavelength
ranges, exposure times, and ecliptic proximity.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from astropy.coordinates import SkyCoord
import astropy.units as u

# Set up logger
logger = logging.getLogger('mast_kbo')

# Import utilities (these would be imported from mast.utils in practice)
# This allows core.py to stand alone if needed
try:
    from mast.utils import (
        is_near_ecliptic, 
        KBO_DETECTION_CONSTANTS, 
        generate_timestamp,
        save_json,
        load_json
    )
except ImportError:
    logger.warning("Unable to import from mast.utils. Some functionality may be limited.")
    
    # Provide minimal implementations for critical functions
    def is_near_ecliptic(ra, dec, max_ecliptic_latitude=5.0):
        """Check if coordinates are close to the ecliptic plane"""
        try:
            from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            ecliptic_coord = coord.transform_to('ecliptic')
            return abs(ecliptic_coord.lat.deg) <= max_ecliptic_latitude
        except Exception as e:
            logger.error(f"Error checking ecliptic proximity: {e}")
            return True  # Default to True to avoid filtering out data
    
    def generate_timestamp():
        """Generate a timestamp string for filenames"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_json(data, filename, indent=2):
        """Save data to JSON file with proper directory creation"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
    
    def load_json(filename):
        """Load data from JSON file with error handling"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {filename}")
    
    # Define KBO detection constants
    KBO_DETECTION_CONSTANTS = {
        # Wavelength range for optimal KBO detection (microns)
        'MIN_WAVELENGTH': 10.0,
        'MAX_WAVELENGTH': 25.0,
        
        # Preferred JWST filters for KBO detection
        'PREFERRED_FILTERS': [
            'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W'
        ],
        
        # Minimum exposure time for KBO detection (seconds)
        'MIN_EXPOSURE_TIME': 300,
        
        # Time constraints for KBO sequences (hours)
        'MIN_SEQUENCE_INTERVAL': 2.0,
        'MAX_SEQUENCE_INTERVAL': 24.0,
        
        # Motion parameters for KBOs
        'TYPICAL_MOTION_RATE': 3.0,  # arcsec/hour
        
        # Minimum field matching radius (arcsec)
        'FIELD_MATCH_RADIUS': 60.0
    }

def filter_by_instrument(observations, include_nircam=False):
    """
    Filter for MIRI observations and optionally NIRCam observations
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    include_nircam : bool
        Whether to include NIRCam observations
        
    Returns:
    --------
    list : Filtered observations
    """
    if not observations:
        logger.warning("No observations to filter by instrument")
        return []

    try:
        if include_nircam:
            instrument_obs = [obs for obs in observations 
                            if 'instrument_name' in obs and 
                            ('MIRI' in obs.get('instrument_name', '') or 
                             'NIRCAM' in obs.get('instrument_name', ''))]
            logger.info(f"MIRI or NIRCam instruments: {len(instrument_obs)}/{len(observations)} observations passed")
        else:
            instrument_obs = [obs for obs in observations 
                            if 'instrument_name' in obs and 'MIRI' in obs.get('instrument_name', '')]
            logger.info(f"MIRI instrument only: {len(instrument_obs)}/{len(observations)} observations passed")
        
        return instrument_obs
    
    except Exception as e:
        logger.error(f"Error filtering by instrument: {e}")
        return observations  # Return original list in case of error

def filter_by_wavelength(observations):
    """
    Filter observations for optimal KBO wavelength range
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
        
    Returns:
    --------
    list : Filtered observations
    """
    if not observations:
        logger.warning("No observations to filter by wavelength")
        return []

    try:
        # Extract constants
        min_wavelength = KBO_DETECTION_CONSTANTS['MIN_WAVELENGTH']
        max_wavelength = KBO_DETECTION_CONSTANTS['MAX_WAVELENGTH']
        preferred_filters = KBO_DETECTION_CONSTANTS['PREFERRED_FILTERS']
        
        wavelength_obs = []
        
        for obs in observations:
            # Check if wavelength information exists
            if 'wavelength_range' in obs and obs['wavelength_range']:
                try:
                    wavelength = obs['wavelength_range']
                    if isinstance(wavelength, (list, tuple, np.ndarray)) and len(wavelength) >= 2:
                        wl_min, wl_max = wavelength[0], wavelength[1]
                        
                        # Convert to microns if needed
                        if wl_min < 1e-4 or wl_max < 1e-4:  # Likely in meters
                            wl_min *= 1e6
                            wl_max *= 1e6
                        
                        if wl_max >= min_wavelength and wl_min <= max_wavelength:
                            wavelength_obs.append(obs)
                            continue  # Skip checking filters
                except (ValueError, TypeError, IndexError):
                    # If we can't parse wavelength, don't exclude based on this
                    pass
            
            # Check filters if available
            if 'filters' in obs:
                filter_name = str(obs.get('filters', ''))
                if any(preferred in filter_name for preferred in preferred_filters):
                    wavelength_obs.append(obs)
                    continue
            
            # As a fallback, check instrument name for MIRI
            if 'instrument_name' in obs and 'MIRI' in obs.get('instrument_name', ''):
                # MIRI operates in the wavelength range we want
                wavelength_obs.append(obs)
        
        logger.info(f"Wavelength range {min_wavelength}-{max_wavelength}μm: {len(wavelength_obs)}/{len(observations)} observations passed")
        return wavelength_obs
    
    except Exception as e:
        logger.error(f"Error filtering by wavelength: {e}")
        return observations  # Return original list in case of error

def filter_by_exposure(observations, min_exposure_time=None):
    """
    Filter for appropriate exposure time range
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    min_exposure_time : float or None
        Minimum exposure time in seconds (overrides default from constants)
        
    Returns:
    --------
    list : Filtered observations
    """
    if not observations:
        logger.warning("No observations to filter by exposure time")
        return []

    try:
        # Extract constants
        if min_exposure_time is None:
            min_exposure_time = KBO_DETECTION_CONSTANTS['MIN_EXPOSURE_TIME']
        
        # Find exposure time field (could be 't_exptime', 'exptime', etc.)
        exposure_field = None
        exposure_fields = ['t_exptime', 'exptime', 'exposure_time']
        
        if observations:
            for field in exposure_fields:
                if field in observations[0]:
                    exposure_field = field
                    break
        
        if not exposure_field:
            logger.warning("No exposure time field found, skipping exposure filter")
            return observations
        
        exposure_obs = [obs for obs in observations 
                       if obs.get(exposure_field, 0) >= min_exposure_time]
        
        logger.info(f"Exposure time ≥{min_exposure_time}s: {len(exposure_obs)}/{len(observations)} observations passed")
        return exposure_obs
    
    except Exception as e:
        logger.error(f"Error filtering by exposure time: {e}")
        return observations  # Return original list in case of error

def filter_by_ecliptic_latitude(observations, max_ecliptic_latitude=5.0):
    """
    Filter observations by proximity to ecliptic plane
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    max_ecliptic_latitude : float
        Maximum absolute ecliptic latitude in degrees
        
    Returns:
    --------
    list : Filtered observations
    """
    if not observations:
        logger.warning("No observations to filter by ecliptic latitude")
        return []

    try:
        # Find coordinate fields (could be 's_ra'/'s_dec', 'ra'/'dec', etc.)
        ra_field, dec_field = None, None
        coordinate_fields = [('s_ra', 's_dec'), ('ra', 'dec'), ('RA', 'DEC')]
        
        if observations:
            for ra_f, dec_f in coordinate_fields:
                if ra_f in observations[0] and dec_f in observations[0]:
                    ra_field, dec_field = ra_f, dec_f
                    break
        
        if not ra_field or not dec_field:
            logger.warning("No coordinate fields found, skipping ecliptic filter")
            return observations
        
        ecliptic_obs = []
        
        for obs in observations:
            try:
                ra = float(obs[ra_field])
                dec = float(obs[dec_field])
                
                if is_near_ecliptic(ra, dec, max_ecliptic_latitude):
                    ecliptic_obs.append(obs)
            except (ValueError, TypeError):
                # If coordinates can't be parsed, keep the observation
                ecliptic_obs.append(obs)
        
        logger.info(f"Ecliptic latitude ≤{max_ecliptic_latitude}°: {len(ecliptic_obs)}/{len(observations)} observations passed")
        return ecliptic_obs
    
    except Exception as e:
        logger.error(f"Error filtering by ecliptic latitude: {e}")
        return observations  # Return original list in case of error

def extract_observations_from_catalog(catalog):
    """
    Extract observations from a catalog in various formats
    
    Parameters:
    -----------
    catalog : dict or list
        Catalog in one of several possible formats
        
    Returns:
    --------
    list : List of observation dictionaries
    """
    observations = []
    
    try:
        if isinstance(catalog, list):
            # List of sequences or observations
            if catalog and 'observations' in catalog[0]:
                # List of sequences
                for sequence in catalog:
                    if 'observations' in sequence:
                        for obs in sequence['observations']:
                            # Add field info to observation
                            obs['field_id'] = sequence.get('field_id', 'unknown')
                            observations.append(obs)
            else:
                # Direct list of observations
                observations = catalog
        elif isinstance(catalog, dict):
            # Dictionary format
            if 'results' in catalog:
                # Combined results format
                for result in catalog['results']:
                    if 'observations' in result:
                        for obs in result['observations']:
                            # Add field info to observation
                            obs['field_id'] = result.get('square_id', 'unknown')
                            observations.append(obs)
            elif 'observations' in catalog:
                # Simple container format
                observations = catalog['observations']
    except Exception as e:
        logger.error(f"Error extracting observations from catalog: {e}")
    
    return observations

def filter_observations(observations, include_nircam=False, 
                      max_ecliptic_latitude=5.0, min_exposure_time=None):
    """
    Apply all filters to a set of observations
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    include_nircam : bool
        Whether to include NIRCam observations
    max_ecliptic_latitude : float
        Maximum absolute ecliptic latitude in degrees
    min_exposure_time : float or None
        Minimum exposure time in seconds
        
    Returns:
    --------
    list : Filtered observations
    """
    logger.info(f"Starting filtering with {len(observations)} observations")
    
    # Apply each filter in sequence
    filtered_obs = observations
    filtered_obs = filter_by_instrument(filtered_obs, include_nircam)
    filtered_obs = filter_by_wavelength(filtered_obs)
    filtered_obs = filter_by_exposure(filtered_obs, min_exposure_time)
    filtered_obs = filter_by_ecliptic_latitude(filtered_obs, max_ecliptic_latitude)
    
    logger.info(f"After all filters: {len(filtered_obs)}/{len(observations)} observations passed")
    return filtered_obs

def filter_catalog(catalog_file, output_file=None, include_nircam=False, 
                  max_ecliptic_latitude=5.0, min_exposure_time=None,
                  min_sequence_interval=None, max_sequence_interval=None):
    """
    Filter a catalog of observations for KBO detection
    
    Parameters:
    -----------
    catalog_file : str
        Path to catalog JSON file
    output_file : str or None
        Path to output filtered catalog (if None, generates default path)
    include_nircam : bool
        Whether to include NIRCam observations
    max_ecliptic_latitude : float
        Maximum absolute ecliptic latitude in degrees
    min_exposure_time : float or None
        Minimum exposure time in seconds
    min_sequence_interval : float or None
        Minimum interval between observations in hours
    max_sequence_interval : float or None
        Maximum interval between observations in hours
        
    Returns:
    --------
    dict : Filtering results including filtered catalog
    """
    # Import sequence functionality
    try:
        from .sequences import (
            group_observations_by_field,
            find_observation_sequences,
            calculate_kbo_motion_parameters,
            score_sequences
        )
    except ImportError:
        logger.error("Could not import sequence functions. Make sure sequences.py is properly installed.")
        return None
    
    # Override KBO constants with specified values
    if min_exposure_time is not None:
        KBO_DETECTION_CONSTANTS['MIN_EXPOSURE_TIME'] = min_exposure_time
    if min_sequence_interval is not None:
        KBO_DETECTION_CONSTANTS['MIN_SEQUENCE_INTERVAL'] = min_sequence_interval
    if max_sequence_interval is not None:
        KBO_DETECTION_CONSTANTS['MAX_SEQUENCE_INTERVAL'] = max_sequence_interval
    
    # Generate output file path if not specified
    if output_file is None:
        timestamp = generate_timestamp()
        output_dir = os.path.dirname(catalog_file)
        basename = os.path.splitext(os.path.basename(catalog_file))[0]
        output_file = os.path.join(output_dir, f"{basename}_filtered_{timestamp}.json")
    
    # Load catalog
    logger.info(f"Loading catalog from {catalog_file}...")
    try:
        catalog = load_json(catalog_file)
    except Exception as e:
        logger.error(f"Error loading catalog: {e}")
        return None
    
    # Extract observations from catalog based on format
    observations = extract_observations_from_catalog(catalog)
    
    logger.info(f"Extracted {len(observations)} observations from catalog")
    
    # Apply filters
    filtered_obs = filter_observations(
        observations, 
        include_nircam=include_nircam,
        max_ecliptic_latitude=max_ecliptic_latitude, 
        min_exposure_time=min_exposure_time
    )
    
    # Group by field and identify sequences
    fields = group_observations_by_field(filtered_obs)
    sequences = find_observation_sequences(fields)
    
    # Calculate motion parameters and score sequences
    sequences = calculate_kbo_motion_parameters(sequences)
    sequences = score_sequences(sequences)
    
    # Prepare results
    results = {
        'timestamp': datetime.now().isoformat(),
        'source_catalog': catalog_file,
        'filter_params': {
            'include_nircam': include_nircam,
            'max_ecliptic_latitude': max_ecliptic_latitude,
            'min_exposure_time': KBO_DETECTION_CONSTANTS['MIN_EXPOSURE_TIME'],
            'min_sequence_interval': KBO_DETECTION_CONSTANTS['MIN_SEQUENCE_INTERVAL'],
            'max_sequence_interval': KBO_DETECTION_CONSTANTS['MAX_SEQUENCE_INTERVAL']
        },
        'stats': {
            'initial_observations': len(observations),
            'filtered_observations': len(filtered_obs),
            'fields': len(fields),
            'sequences': len(sequences)
        },
        'sequences': sequences
    }
    
    # Save results
    logger.info(f"Saving filtered catalog to {output_file}")
    try:
        save_json(results, output_file)
    except Exception as e:
        logger.error(f"Error saving filtered catalog: {e}")
    
    return results

def quick_filter(observations, strict=False, include_nircam=False):
    """
    Quick filter of observations for KBO candidates with preset parameters
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    strict : bool
        Whether to use strict filtering parameters
    include_nircam : bool
        Whether to include NIRCam observations
        
    Returns:
    --------
    list : Filtered observations
    """
    # Set parameters based on strictness
    if strict:
        ecliptic_latitude = 3.0  # More restrictive
        min_exposure = 600  # Longer exposures
    else:
        ecliptic_latitude = 5.0  # Standard
        min_exposure = 300  # Standard
    
    # Apply filters
    filtered_obs = filter_observations(
        observations,
        include_nircam=include_nircam,
        max_ecliptic_latitude=ecliptic_latitude,
        min_exposure_time=min_exposure
    )
    
    return filtered_obs