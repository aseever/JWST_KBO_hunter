"""
mast/filter.py - Filter JWST observations for KBO detection

This module filters JWST observations to identify the most promising candidates
for KBO detection, focusing on appropriate wavelengths, exposure times, time 
sequences, and proximity to the ecliptic plane.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

# Import utilities
from mast.utils import (
    is_near_ecliptic, 
    KBO_DETECTION_CONSTANTS, 
    generate_timestamp,
    save_json,
    load_json
)

# Set up logger
logger = logging.getLogger('mast_kbo')

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

def filter_by_exposure(observations):
    """
    Filter for appropriate exposure time range
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
        
    Returns:
    --------
    list : Filtered observations
    """
    # Extract constants
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

def group_observations_by_field(observations, max_radius=None):
    """
    Group observations by field (similar coordinates)
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    max_radius : float or None
        Maximum radius in arcseconds to consider same field
        
    Returns:
    --------
    dict : Dictionary mapping field_id to list of observations
    """
    if max_radius is None:
        max_radius = KBO_DETECTION_CONSTANTS['FIELD_MATCH_RADIUS']
    
    # Find coordinate fields
    ra_field, dec_field = None, None
    coordinate_fields = [('s_ra', 's_dec'), ('ra', 'dec'), ('RA', 'DEC')]
    
    if observations:
        for ra_f, dec_f in coordinate_fields:
            if ra_f in observations[0] and dec_f in observations[0]:
                ra_field, dec_field = ra_f, dec_f
                break
    
    if not ra_field or not dec_field:
        logger.warning("No coordinate fields found, grouping by original field/sequence")
        # Try to group by original field_id if available
        fields = {}
        for obs in observations:
            field_id = obs.get('field_id', 'unknown')
            if field_id not in fields:
                fields[field_id] = []
            fields[field_id].append(obs)
        return fields
    
    # Group by coordinates
    fields = {}
    field_index = 0
    
    for obs in observations:
        try:
            ra = float(obs[ra_field])
            dec = float(obs[dec_field])
            
            obs_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
            field_match = False
            
            for field_id, field_data in fields.items():
                field_coords = field_data['center_coord']
                separation = obs_coord.separation(field_coords).arcsec
                
                if separation <= max_radius:
                    # Add to existing field
                    field_data['observations'].append(obs)
                    field_match = True
                    break
            
            if not field_match:
                # Create new field
                field_id = f"field_{field_index}"
                fields[field_id] = {
                    'center_coord': obs_coord,
                    'center_ra': ra,
                    'center_dec': dec,
                    'observations': [obs]
                }
                field_index += 1
        
        except (ValueError, TypeError):
            # If coordinates can't be parsed, put in unknown field
            if 'unknown' not in fields:
                fields['unknown'] = {
                    'center_coord': None,
                    'center_ra': None,
                    'center_dec': None,
                    'observations': []
                }
            fields['unknown']['observations'].append(obs)
    
    # Count fields
    multi_obs_fields = {field_id: data for field_id, data in fields.items() 
                        if len(data['observations']) > 1}
    
    logger.info(f"Field grouping: Found {len(fields)} distinct fields")
    logger.info(f"Fields with multiple observations: {len(multi_obs_fields)}/{len(fields)}")
    
    # Convert to simpler format
    result = {}
    for field_id, data in fields.items():
        result[field_id] = data['observations']
    
    return result

def find_observation_sequences(fields, min_sequence_interval=None, max_sequence_interval=None, min_sequence_length=2):
    """
    Find sequences of observations suitable for KBO detection
    
    Parameters:
    -----------
    fields : dict
        Dictionary mapping field_id to list of observations
    min_sequence_interval : float or None
        Minimum time interval between observations in hours
    max_sequence_interval : float or None
        Maximum time interval between observations in hours
    min_sequence_length : int
        Minimum number of observations in a sequence
        
    Returns:
    --------
    list : List of sequence dictionaries
    """
    # Use default values if not specified
    if min_sequence_interval is None:
        min_sequence_interval = KBO_DETECTION_CONSTANTS['MIN_SEQUENCE_INTERVAL']
    if max_sequence_interval is None:
        max_sequence_interval = KBO_DETECTION_CONSTANTS['MAX_SEQUENCE_INTERVAL']
    
    sequences = []
    
    # Find time field (could be 't_min', 'date_obs', etc.)
    time_fields = ['t_min', 'date_obs', 'observation_time', 'obs_time']
    
    for field_id, observations in fields.items():
        if len(observations) < min_sequence_length:
            continue
        
        # Try to find a usable time field
        time_field = None
        for field in time_fields:
            if observations and field in observations[0]:
                time_field = field
                break
        
        if not time_field:
            logger.warning(f"No time field found for field {field_id}, skipping sequence detection")
            continue
        
        # Sort observations by time
        obs_with_time = []
        for obs in observations:
            if time_field not in obs:
                continue
                
            try:
                time_value = obs[time_field]
                if isinstance(time_value, str):
                    time_obj = Time(time_value, format='isot')
                    time_mjd = time_obj.mjd
                else:
                    # Assume it's already in MJD format
                    time_mjd = float(time_value)
                
                obs_with_time.append((obs, time_mjd))
            except (ValueError, TypeError):
                # Skip observations with invalid time
                continue
        
        # Skip if not enough observations with valid time
        if len(obs_with_time) < min_sequence_length:
            continue
            
        # Sort by time
        obs_with_time.sort(key=lambda x: x[1])
        
        # Find sequences with appropriate time intervals
        current_sequence = [obs_with_time[0]]
        
        for i in range(1, len(obs_with_time)):
            prev_obs, prev_time = current_sequence[-1]
            current_obs, current_time = obs_with_time[i]
            
            # Calculate time difference in hours
            time_diff_hours = (current_time - prev_time) * 24.0
            
            # Check if this observation continues the sequence
            if min_sequence_interval <= time_diff_hours <= max_sequence_interval:
                current_sequence.append((current_obs, current_time))
            else:
                # If sequence is long enough, save it and start a new one
                if len(current_sequence) >= min_sequence_length:
                    sequence_obs = [item[0] for item in current_sequence]
                    start_time = Time(current_sequence[0][1], format='mjd').iso
                    end_time = Time(current_sequence[-1][1], format='mjd').iso
                    duration_hours = (current_sequence[-1][1] - current_sequence[0][1]) * 24.0
                    
                    sequences.append({
                        'field_id': field_id,
                        'num_observations': len(sequence_obs),
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration_hours': duration_hours,
                        'observations': sequence_obs
                    })
                
                # Start a new sequence
                current_sequence = [(current_obs, current_time)]
        
        # Check if last sequence is valid
        if len(current_sequence) >= min_sequence_length:
            sequence_obs = [item[0] for item in current_sequence]
            start_time = Time(current_sequence[0][1], format='mjd').iso
            end_time = Time(current_sequence[-1][1], format='mjd').iso
            duration_hours = (current_sequence[-1][1] - current_sequence[0][1]) * 24.0
            
            sequences.append({
                'field_id': field_id,
                'num_observations': len(sequence_obs),
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': duration_hours,
                'observations': sequence_obs
            })
    
    # Sort sequences by number of observations (descending)
    sequences.sort(key=lambda x: x['num_observations'], reverse=True)
    
    logger.info(f"Sequence identification: Found {len(sequences)} valid observation sequences")
    if sequences:
        logger.info(f"  - Longest sequence: {sequences[0]['num_observations']} observations over {sequences[0]['duration_hours']:.1f} hours")
    
    return sequences

def calculate_kbo_motion_parameters(sequences):
    """
    Calculate expected KBO motion parameters for each sequence
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
        
    Returns:
    --------
    list : Sequences with added motion parameters
    """
    typical_motion_rate = KBO_DETECTION_CONSTANTS['TYPICAL_MOTION_RATE']  # arcsec/hour
    
    for sequence in sequences:
        # Calculate expected KBO motion for this sequence
        duration_hours = sequence.get('duration_hours', 0)
        
        if duration_hours > 0:
            # Total expected motion in arcseconds
            expected_motion_arcsec = typical_motion_rate * duration_hours
            
            # Add to sequence
            sequence['expected_motion_arcsec'] = expected_motion_arcsec
            
            # Estimate distance based on motion rate (inverse relationship)
            # AU ≈ 4.74 / arcsec_per_hour
            if typical_motion_rate > 0:
                approx_distance_au = 4.74 / typical_motion_rate
                sequence['approx_distance_au'] = approx_distance_au
    
    return sequences

def score_sequences(sequences):
    """
    Score sequences for KBO detection potential
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
        
    Returns:
    --------
    list : Sequences with added scores
    """
    # Ideal parameters for KBO detection
    ideal_num_obs = 5  # Ideal number of observations
    ideal_duration = 12.0  # Ideal duration in hours
    
    for sequence in sequences:
        # Start with base score
        score = 0.5
        
        # Score by number of observations (more is better, up to 5)
        num_obs = sequence['num_observations']
        if num_obs >= ideal_num_obs:
            score += 0.25  # Max bonus
        else:
            # Partial bonus for fewer observations
            score += 0.25 * (num_obs / ideal_num_obs)
        
        # Score by duration (closer to ideal is better)
        duration = sequence.get('duration_hours', 0)
        if duration > 0:
            duration_ratio = min(duration, 2 * ideal_duration) / ideal_duration
            # Score higher for durations closer to ideal
            duration_score = 0.25 * (1.0 - abs(duration_ratio - 1.0))
            score += duration_score
        
        # Bonus for observations near ecliptic plane
        if 'center_dec' in sequence and abs(sequence.get('center_dec', 90)) < 5.0:
            score += 0.1
        
        # Ensure score is between 0 and 1
        sequence['kbo_score'] = max(0.0, min(1.0, score))
    
    # Sort by score
    sequences.sort(key=lambda x: x['kbo_score'], reverse=True)
    
    return sequences

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
    observations = []
    
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
    
    logger.info(f"Extracted {len(observations)} observations from catalog")
    
    # Apply filters
    filtered_obs = observations
    filtered_obs = filter_by_instrument(filtered_obs, include_nircam)
    filtered_obs = filter_by_wavelength(filtered_obs)
    filtered_obs = filter_by_exposure(filtered_obs)
    filtered_obs = filter_by_ecliptic_latitude(filtered_obs, max_ecliptic_latitude)
    
    logger.info(f"After all filters: {len(filtered_obs)}/{len(observations)} observations passed")
    
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

def generate_filter_visualizations(results, output_dir=None):
    """
    Generate visualizations of the filtering results
    
    Parameters:
    -----------
    results : dict
        Results from filter_catalog
    output_dir : str or None
        Output directory for visualizations
        
    Returns:
    --------
    list : Paths to generated visualization files
    """
    if not results or 'sequences' not in results:
        logger.error("Invalid results for visualization")
        return []
    
    # Determine output directory
    if output_dir is None:
        if 'source_catalog' in results:
            output_dir = os.path.join(os.path.dirname(results['source_catalog']), 'visualizations')
        else:
            output_dir = 'visualizations'
    
    os.makedirs(output_dir, exist_ok=True)
    
    visualization_files = []
    
    # 1. Sequence length histogram
    if results['sequences']:
        plt.figure(figsize=(10, 6))
        seq_lengths = [seq['num_observations'] for seq in results['sequences']]
        plt.hist(seq_lengths, bins=range(2, max(seq_lengths) + 2), alpha=0.7, 
                 color='steelblue', edgecolor='black')
        plt.xlabel('Number of Observations in Sequence')
        plt.ylabel('Count')
        plt.title('Observation Sequence Lengths')
        plt.xticks(range(2, max(seq_lengths) + 1))
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(output_dir, 'sequence_lengths.png')
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        visualization_files.append(output_file)
        logger.info(f"Generated sequence length visualization: {output_file}")
    
    # 2. Sequence duration histogram
    if results['sequences']:
        plt.figure(figsize=(10, 6))
        durations = [seq.get('duration_hours', 0) for seq in results['sequences']]
        plt.hist(durations, bins=10, alpha=0.7, color='mediumseagreen', edgecolor='black')
        plt.xlabel('Sequence Duration (hours)')
        plt.ylabel('Count')
        plt.title('Observation Sequence Durations')
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(output_dir, 'sequence_durations.png')
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        visualization_files.append(output_file)
        logger.info(f"Generated sequence duration visualization: {output_file}")
    
    # 3. Sequence scores
    if results['sequences']:
        plt.figure(figsize=(10, 6))
        scores = [seq.get('kbo_score', 0) for seq in results['sequences']]
        plt.hist(scores, bins=10, alpha=0.7, color='darkorange', edgecolor='black')
        plt.xlabel('KBO Detection Score')
        plt.ylabel('Count')
        plt.title('Sequence KBO Detection Scores')
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(output_dir, 'sequence_scores.png')
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        visualization_files.append(output_file)
        logger.info(f"Generated sequence score visualization: {output_file}")
    
    # 4. Filter effectiveness pie chart
    stats = results.get('stats', {})
    if 'initial_observations' in stats and 'filtered_observations' in stats:
        plt.figure(figsize=(8, 8))
        
        # Create data for pie chart
        initial = stats['initial_observations']
        filtered = stats['filtered_observations']
        removed = initial - filtered
        
        sizes = [filtered, removed]
        labels = [f'Passed Filters\n({filtered} obs)', f'Filtered Out\n({removed} obs)']
        colors = ['mediumseagreen', 'lightgray']
        explode = (0.1, 0)  # Explode the first slice
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Filter Effectiveness')
        
        output_file = os.path.join(output_dir, 'filter_effectiveness.png')
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        visualization_files.append(output_file)
        logger.info(f"Generated filter effectiveness visualization: {output_file}")
    
    return visualization_files