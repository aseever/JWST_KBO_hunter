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
    This function now passes through all observations as instrument filtering
    is handled at the MAST query level
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    include_nircam : bool
        Parameter kept for backward compatibility, no longer used
        
    Returns:
    --------
    list : Same observations list (no filtering)
    """
    if not observations:
        logger.warning("No observations to filter by instrument")
        return []

    try:
        # Log observation count but don't filter
        logger.info(f"Instrument filtering disabled (handled by MAST query): passing {len(observations)} observations")
        
        # Report instrument types found (for informational purposes)
        instrument_types = {}
        for obs in observations:
            inst_name = obs.get('instrument_name', 'unknown')
            instrument_types[inst_name] = instrument_types.get(inst_name, 0) + 1
        
        if instrument_types:
            logger.info(f"Instrument types found: {instrument_types}")
        
        # Return all observations without filtering
        return observations
    
    except Exception as e:
        logger.error(f"Error in instrument type logging: {e}")
        return observations

def filter_by_wavelength(observations):
    """
    This function now passes through all observations as wavelength filtering
    is handled at the MAST query level
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
        
    Returns:
    --------
    list : Same observations list (no filtering)
    """
    if not observations:
        logger.warning("No observations to filter by wavelength")
        return []

    try:
        logger.info(f"Wavelength filtering disabled (handled by MAST query): passing {len(observations)} observations")
        return observations
    
    except Exception as e:
        logger.error(f"Error in wavelength filtering: {e}")
        return observations

def filter_by_exposure(observations, min_exposure_time=None):
    """
    This function now passes through all observations as exposure filtering
    is handled at the MAST query level
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    min_exposure_time : float or None
        Parameter kept for backward compatibility, no longer used
        
    Returns:
    --------
    list : Same observations list (no filtering)
    """
    if not observations:
        logger.warning("No observations to filter by exposure time")
        return []

    try:
        logger.info(f"Exposure filtering disabled (handled by MAST query): passing {len(observations)} observations")
        return observations
    
    except Exception as e:
        logger.error(f"Error in exposure filtering: {e}")
        return observations

def filter_by_ecliptic_latitude(observations, max_ecliptic_latitude=5.0):
    """
    This function now passes through all observations as ecliptic latitude filtering
    is handled at the MAST query level
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    max_ecliptic_latitude : float
        Parameter kept for backward compatibility, no longer used
        
    Returns:
    --------
    list : Same observations list (no filtering)
    """
    if not observations:
        logger.warning("No observations to filter by ecliptic latitude")
        return []

    try:
        logger.info(f"Ecliptic latitude filtering disabled (handled by MAST query): passing {len(observations)} observations")
        return observations
    
    except Exception as e:
        logger.error(f"Error in ecliptic latitude filtering: {e}")
        return observations

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
    Apply all filters to a set of observations - currently disabled as filtering
    is handled at the MAST query level
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    include_nircam : bool
        Parameter kept for backward compatibility
    max_ecliptic_latitude : float
        Parameter kept for backward compatibility
    min_exposure_time : float or None
        Parameter kept for backward compatibility
        
    Returns:
    --------
    list : Same observations list (no filtering)
    """
    logger.info(f"Starting observation processing with {len(observations)} observations")
    logger.info(f"All filtering disabled (handled by MAST query): passing all observations")
    
    # Log some key stats about the data 
    if observations and len(observations) > 0:
        fields = list(observations[0].keys())
        logger.info(f"Fields available in data: {', '.join(fields)}")
    
    # Call the filter functions but they now just pass through data
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
        Whether to include NIRCam observations - kept for backward compatibility
    max_ecliptic_latitude : float
        Maximum absolute ecliptic latitude in degrees - kept for backward compatibility
    min_exposure_time : float or None
        Minimum exposure time in seconds - kept for backward compatibility
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
    
    # Apply filters (now disabled - passing through all data)
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
    This is now disabled as filtering is handled at the MAST query level
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    strict : bool
        Parameter kept for backward compatibility
    include_nircam : bool
        Parameter kept for backward compatibility
        
    Returns:
    --------
    list : Same observations list (no filtering)
    """
    logger.info(f"Quick filtering disabled: passing all {len(observations)} observations")
    return observations