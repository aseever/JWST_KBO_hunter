"""
mast/filter/analysis.py - Advanced analysis features for KBO detection

This module provides advanced analysis capabilities for KBO detection,
including sensitivity analysis, near-miss detection, and observation
quality assessment. These functions help identify promising KBO candidates
and evaluate detection limitations.
"""

import logging
import numpy as np
from datetime import datetime
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
import astropy.units as u

# Set up logger
logger = logging.getLogger('mast_kbo')

# Import utilities
try:
    from mast.utils import (
        is_near_ecliptic, 
        KBO_DETECTION_CONSTANTS
    )
except ImportError:
    logger.warning("Unable to import from mast.utils. Using default constants.")
    # Define default KBO detection constants
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

def analyze_sequence_coverage(sequences):
    """
    Analyze the spatial and temporal coverage of sequences
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
        
    Returns:
    --------
    dict : Coverage analysis results
    """
    if not sequences:
        logger.info("No sequences to analyze coverage")
        return {
            'spatial_coverage': {
                'min_ra': None,
                'max_ra': None,
                'min_dec': None,
                'max_dec': None,
                'area_sq_deg': 0
            },
            'temporal_coverage': {
                'min_date': None,
                'max_date': None,
                'span_days': 0
            },
            'ecliptic_coverage': 0.0
        }
    
    try:
        # Extract coordinates and times
        coords = []
        mjd_times = []
        
        for seq in sequences:
            if 'center_ra' in seq and 'center_dec' in seq and seq['center_ra'] is not None and seq['center_dec'] is not None:
                coords.append((seq['center_ra'], seq['center_dec']))
            
            if 'start_time' in seq:
                try:
                    mjd_times.append(Time(seq['start_time'], format='isot').mjd)
                except ValueError:
                    pass
                    
            if 'end_time' in seq:
                try:
                    mjd_times.append(Time(seq['end_time'], format='isot').mjd)
                except ValueError:
                    pass
        
        # Spatial coverage
        spatial_coverage = {}
        if coords:
            ra_values = [c[0] for c in coords]
            dec_values = [c[1] for c in coords]
            
            # Handle RA wrap-around
            if max(ra_values) - min(ra_values) > 180:
                # Adjust RAs that are > 180 degrees apart
                ra_values = [ra if ra < 180 else ra - 360 for ra in ra_values]
            
            min_ra = min(ra_values)
            max_ra = max(ra_values)
            min_dec = min(dec_values)
            max_dec = max(dec_values)
            
            # Estimate covered area in square degrees
            # This is approximate - assumes a rectangular region
            ra_span = max_ra - min_ra
            dec_span = max_dec - min_dec
            
            # Adjust for spherical coordinates
            avg_dec = (min_dec + max_dec) / 2
            area_sq_deg = ra_span * dec_span * np.cos(np.radians(avg_dec))
            
            spatial_coverage = {
                'min_ra': min_ra,
                'max_ra': max_ra,
                'min_dec': min_dec,
                'max_dec': max_dec,
                'ra_span': ra_span,
                'dec_span': dec_span,
                'area_sq_deg': abs(area_sq_deg)
            }
            
            # Calculate ecliptic coverage
            ecliptic_coverage = calculate_ecliptic_coverage(min_ra, max_ra, min_dec, max_dec)
        else:
            spatial_coverage = {
                'min_ra': None,
                'max_ra': None,
                'min_dec': None,
                'max_dec': None,
                'ra_span': 0,
                'dec_span': 0,
                'area_sq_deg': 0
            }
            ecliptic_coverage = 0.0
        
        # Temporal coverage
        temporal_coverage = {}
        if mjd_times:
            min_mjd = min(mjd_times)
            max_mjd = max(mjd_times)
            span_days = max_mjd - min_mjd
            
            min_date = Time(min_mjd, format='mjd').iso.split('T')[0]
            max_date = Time(max_mjd, format='mjd').iso.split('T')[0]
            
            temporal_coverage = {
                'min_date': min_date,
                'max_date': max_date,
                'min_mjd': min_mjd,
                'max_mjd': max_mjd,
                'span_days': span_days
            }
        else:
            temporal_coverage = {
                'min_date': None,
                'max_date': None,
                'min_mjd': None,
                'max_mjd': None,
                'span_days': 0
            }
        
        # Combine results
        results = {
            'spatial_coverage': spatial_coverage,
            'temporal_coverage': temporal_coverage,
            'ecliptic_coverage': ecliptic_coverage
        }
        
        logger.info(f"Analyzed coverage of {len(sequences)} sequences")
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing sequence coverage: {e}")
        return {
            'error': str(e),
            'spatial_coverage': {},
            'temporal_coverage': {}
        }

def calculate_ecliptic_coverage(min_ra, max_ra, min_dec, max_dec):
    """
    Calculate the approximate coverage of the ecliptic plane
    
    Parameters:
    -----------
    min_ra, max_ra : float
        RA limits in degrees
    min_dec, max_dec : float
        Dec limits in degrees
        
    Returns:
    --------
    float : Approximate ecliptic coverage (0-1)
    """
    try:
        # Generate points along the ecliptic
        # We'll sample the full ecliptic and see what fraction falls within our region
        num_samples = 360  # One sample per degree of longitude
        
        ecliptic_lon = np.linspace(0, 360, num_samples) * u.deg
        ecliptic_lat = np.zeros(num_samples) * u.deg
        
        # Convert to equatorial coordinates
        ecliptic_coords = SkyCoord(
            lon=ecliptic_lon,
            lat=ecliptic_lat,
            frame=GeocentricTrueEcliptic
        )
        
        equatorial_coords = ecliptic_coords.transform_to('icrs')
        
        # Count points inside our region
        ra_values = equatorial_coords.ra.deg
        dec_values = equatorial_coords.dec.deg
        
        # Handle RA wrap-around in the same way as the input
        if max_ra - min_ra > 180:
            ra_values = np.array([ra if ra < 180 else ra - 360 for ra in ra_values])
        
        # Count points in the region
        points_in_region = sum(
            (min_ra <= ra <= max_ra) and (min_dec <= dec <= max_dec)
            for ra, dec in zip(ra_values, dec_values)
        )
        
        # Calculate coverage fraction
        coverage = points_in_region / num_samples
        
        return coverage
    
    except Exception as e:
        logger.error(f"Error calculating ecliptic coverage: {e}")
        return 0.0

def estimate_detection_sensitivity(sequences):
    """
    Estimate sensitivity of sequences for KBO detection
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
        
    Returns:
    --------
    dict : Sensitivity analysis results
    """
    if not sequences:
        logger.info("No sequences to estimate detection sensitivity")
        return {'sensitivity_by_distance': {}}
    
    try:
        # Sensitivity thresholds based on sequence properties
        # For each distance (AU), the min_score required for detection
        sensitivity_thresholds = {
            30: 0.5,    # 30 AU (Neptune)
            40: 0.6,    # 40 AU (typical KBO)
            50: 0.75,   # 50 AU (distant KBO)
            70: 0.85,   # 70 AU (very distant KBO)
            100: 0.95,  # 100 AU (extreme distant object)
        }
        
        # Count detectable sequences at each distance
        sensitivity_by_distance = {}
        
        for distance, min_score in sensitivity_thresholds.items():
            detectable = [seq for seq in sequences if seq.get('kbo_score', 0) >= min_score]
            sensitivity_by_distance[distance] = {
                'detectable_count': len(detectable),
                'total_count': len(sequences),
                'percentage': len(detectable) / len(sequences) * 100 if sequences else 0
            }
        
        # Calculate distance limits for each sequence
        for seq in sequences:
            score = seq.get('kbo_score', 0)
            
            # Find the maximum distance at which this sequence could detect
            max_detection_distance = 30  # Default to Neptune distance
            
            for distance, min_score in sorted(sensitivity_thresholds.items()):
                if score >= min_score:
                    max_detection_distance = distance
                else:
                    break
            
            seq['max_detection_distance_au'] = max_detection_distance
        
        # Overall sensitivity rating
        best_sequences = [seq for seq in sequences if seq.get('kbo_score', 0) >= 0.8]
        good_sequences = [seq for seq in sequences if 0.6 <= seq.get('kbo_score', 0) < 0.8]
        
        if len(best_sequences) >= 3:
            overall_rating = "Excellent"
        elif len(best_sequences) >= 1 or len(good_sequences) >= 3:
            overall_rating = "Good"
        elif len(good_sequences) >= 1:
            overall_rating = "Fair"
        else:
            overall_rating = "Poor"
        
        results = {
            'sensitivity_by_distance': sensitivity_by_distance,
            'overall_rating': overall_rating,
            'best_sequences_count': len(best_sequences),
            'good_sequences_count': len(good_sequences)
        }
        
        logger.info(f"Estimated detection sensitivity for {len(sequences)} sequences")
        logger.info(f"Overall sensitivity rating: {overall_rating}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error estimating detection sensitivity: {e}")
        return {'error': str(e), 'sensitivity_by_distance': {}}

def analyze_near_misses(sequences, observations, max_gap_hours=48.0, spatial_threshold_deg=0.5):
    """
    Analyze near misses in the observation data - sequences that almost, but don't quite,
    meet the criteria for KBO detection
    
    Parameters:
    -----------
    sequences : list
        List of identified sequences
    observations : list
        List of all observations
    max_gap_hours : float
        Maximum time gap to consider for near misses
    spatial_threshold_deg : float
        Maximum spatial separation to consider for near misses
        
    Returns:
    --------
    dict : Near-miss analysis results
    """
    if not sequences or not observations:
        logger.info("Insufficient data to analyze near misses")
        return {'near_misses': []}
    
    try:
        near_misses = []
        
        # Group all observations by field
        try:
            from .sequences import group_observations_by_field
            fields = group_observations_by_field(observations)
        except ImportError:
            logger.warning("Could not import group_observations_by_field, using simplified version")
            fields = {}
            
            # Simple grouping by coordinates
            for obs in observations:
                # Find coordinate fields (could be 's_ra'/'s_dec', 'ra'/'dec', etc.)
                ra_field, dec_field = None, None
                coordinate_fields = [('s_ra', 's_dec'), ('ra', 'dec'), ('RA', 'DEC')]
                
                for ra_f, dec_f in coordinate_fields:
                    if ra_f in obs and dec_f in obs:
                        ra_field, dec_field = ra_f, dec_f
                        break
                
                if not ra_field or not dec_field:
                    continue
                
                try:
                    ra = float(obs[ra_field])
                    dec = float(obs[dec_field])
                    field_key = f"{ra:.1f}_{dec:.1f}"
                    
                    if field_key not in fields:
                        fields[field_key] = []
                    
                    fields[field_key].append(obs)
                except (ValueError, TypeError):
                    continue
        
        # Check for observations that could complete existing sequences
        for seq in sequences:
            if 'center_ra' not in seq or 'center_dec' not in seq:
                continue
                
            ref_coord = SkyCoord(ra=seq['center_ra'], dec=seq['center_dec'], unit=(u.deg, u.deg))
            
            # Get start and end times of sequence
            try:
                seq_start = Time(seq['start_time'], format='isot')
                seq_end = Time(seq['end_time'], format='isot')
            except (ValueError, TypeError):
                continue
            
            # Look for observations that could extend this sequence
            for field_id, field_obs in fields.items():
                # Skip if this field was already included in the sequence
                if field_id == seq.get('field_id'):
                    continue
                
                # Check spatial proximity
                for obs in field_obs:
                    # Find coordinate fields
                    ra_field, dec_field = None, None
                    coordinate_fields = [('s_ra', 's_dec'), ('ra', 'dec'), ('RA', 'DEC')]
                    
                    for ra_f, dec_f in coordinate_fields:
                        if ra_f in obs and dec_f in obs:
                            ra_field, dec_field = ra_f, dec_f
                            break
                    
                    if not ra_field or not dec_field:
                        continue
                    
                    try:
                        ra = float(obs[ra_field])
                        dec = float(obs[dec_field])
                        
                        obs_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
                        separation = ref_coord.separation(obs_coord).deg
                        
                        # If spatially close enough
                        if separation <= spatial_threshold_deg:
                            # Check temporal proximity
                            time_fields = ['t_min', 'date_obs', 'observation_time', 'obs_time']
                            time_field = None
                            
                            for field in time_fields:
                                if field in obs:
                                    time_field = field
                                    break
                            
                            if not time_field:
                                continue
                            
                            try:
                                obs_time = Time(obs[time_field], format='isot')
                                
                                # Calculate time gaps
                                pre_gap_hours = (obs_time - seq_start).to_value('hour')
                                post_gap_hours = (seq_end - obs_time).to_value('hour')
                                
                                # If within time threshold (but outside sequence)
                                if (pre_gap_hours < 0 and abs(pre_gap_hours) <= max_gap_hours) or \
                                   (post_gap_hours < 0 and abs(post_gap_hours) <= max_gap_hours):
                                    
                                    near_miss = {
                                        'sequence_id': seq.get('field_id', 'unknown'),
                                        'sequence_score': seq.get('kbo_score', 0),
                                        'observation': obs,
                                        'spatial_separation_deg': separation,
                                        'would_extend': True
                                    }
                                    
                                    if pre_gap_hours < 0:
                                        near_miss['time_gap_hours'] = abs(pre_gap_hours)
                                        near_miss['position'] = 'before'
                                    else:
                                        near_miss['time_gap_hours'] = abs(post_gap_hours)
                                        near_miss['position'] = 'after'
                                    
                                    near_misses.append(near_miss)
                            except (ValueError, TypeError):
                                continue
                    except (ValueError, TypeError):
                        continue
        
        # Also look for pairs of observations that almost form a sequence
        # (This is more complex and implementation is abbreviated)
        identified_pairs = []
        
        for field_id, field_obs in fields.items():
            if len(field_obs) < 2:
                continue
                
            # Find time field
            time_fields = ['t_min', 'date_obs', 'observation_time', 'obs_time']
            time_field = None
            
            for field in time_fields:
                if field_obs and field in field_obs[0]:
                    time_field = field
                    break
            
            if not time_field:
                continue
            
            # Check pairs of observations
            for i, obs1 in enumerate(field_obs):
                for j, obs2 in enumerate(field_obs[i+1:], i+1):
                    try:
                        time1 = Time(obs1[time_field], format='isot')
                        time2 = Time(obs2[time_field], format='isot')
                        
                        gap_hours = abs((time2 - time1).to_value('hour'))
                        
                        # If gap is just outside sequence interval
                        min_interval = KBO_DETECTION_CONSTANTS['MIN_SEQUENCE_INTERVAL']
                        max_interval = KBO_DETECTION_CONSTANTS['MAX_SEQUENCE_INTERVAL']
                        
                        too_short = gap_hours < min_interval and gap_hours > min_interval * 0.5
                        too_long = gap_hours > max_interval and gap_hours < max_interval * 1.5
                        
                        if too_short or too_long:
                            pair_key = tuple(sorted([i, j]))
                            if pair_key not in identified_pairs:
                                identified_pairs.append(pair_key)
                                
                                near_miss = {
                                    'sequence_id': f"{field_id}_potential",
                                    'observation1': obs1,
                                    'observation2': obs2,
                                    'time_gap_hours': gap_hours,
                                    'would_extend': False
                                }
                                
                                if too_short:
                                    near_miss['issue'] = 'gap_too_short'
                                else:
                                    near_miss['issue'] = 'gap_too_long'
                                
                                near_misses.append(near_miss)
                    except (ValueError, TypeError):
                        continue
        
        # Sort by potential value (combination of sequence score and spatial proximity)
        for miss in near_misses:
            if 'sequence_score' in miss:
                # Higher score is better
                score_factor = miss['sequence_score']
            else:
                score_factor = 0.5  # Default for potential sequences
                
            if 'spatial_separation_deg' in miss:
                # Lower separation is better
                separation_factor = 1.0 - (miss['spatial_separation_deg'] / spatial_threshold_deg)
            else:
                separation_factor = 0.5  # Default
                
            if 'time_gap_hours' in miss:
                # Lower gap is better
                gap_factor = 1.0 - (miss['time_gap_hours'] / max_gap_hours)
            else:
                gap_factor = 0.5  # Default
                
            # Combine factors
            miss['value_score'] = (score_factor + separation_factor + gap_factor) / 3
        
        # Sort by value score
        near_misses.sort(key=lambda x: x.get('value_score', 0), reverse=True)
        
        logger.info(f"Identified {len(near_misses)} near miss opportunities")
        
        return {
            'near_misses': near_misses,
            'count': len(near_misses)
        }
    
    except Exception as e:
        logger.error(f"Error analyzing near misses: {e}")
        return {'error': str(e), 'near_misses': []}

def evaluate_filter_parameters(observations, test_params=None, num_iterations=5):
    """
    Evaluate different filtering parameters to optimize results
    
    Parameters:
    -----------
    observations : list
        List of all observations
    test_params : list or None
        List of parameter sets to test (if None, uses defaults)
    num_iterations : int
        Number of iterations for each parameter set
        
    Returns:
    --------
    dict : Evaluation results
    """
    if not observations:
        logger.info("No observations to evaluate filter parameters")
        return {'parameter_tests': []}
    
    try:
        # Import necessary functions
        try:
            from .core import filter_observations
            from .sequences import group_observations_by_field, find_observation_sequences
        except ImportError:
            logger.error("Could not import required functions for filter evaluation")
            return {'error': 'Missing required functions', 'parameter_tests': []}
        
        # Default parameter sets to test if none provided
        if test_params is None:
            test_params = [
                {
                    'name': 'baseline',
                    'include_nircam': False,
                    'max_ecliptic_latitude': 5.0,
                    'min_exposure_time': 300,
                    'min_sequence_interval': 2.0,
                    'max_sequence_interval': 24.0
                },
                {
                    'name': 'strict',
                    'include_nircam': False,
                    'max_ecliptic_latitude': 3.0,
                    'min_exposure_time': 500,
                    'min_sequence_interval': 3.0,
                    'max_sequence_interval': 18.0
                },
                {
                    'name': 'lenient',
                    'include_nircam': True,
                    'max_ecliptic_latitude': 8.0,
                    'min_exposure_time': 200,
                    'min_sequence_interval': 1.0,
                    'max_sequence_interval': 36.0
                }
            ]
        
        results = []
        
        # Test each parameter set
        for params in test_params:
            logger.info(f"Testing filter parameters: {params['name']}")
            
            # Create parameter variations for Monte Carlo approach
            param_variations = [params]
            
            # Add variations if more than one iteration requested
            if num_iterations > 1:
                for _ in range(num_iterations - 1):
                    # Create a variation with small random adjustments
                    variation = {
                        'name': f"{params['name']}_var{_+1}",
                        'include_nircam': params['include_nircam'],
                        'max_ecliptic_latitude': params['max_ecliptic_latitude'] * random.uniform(0.9, 1.1),
                        'min_exposure_time': params['min_exposure_time'] * random.uniform(0.9, 1.1),
                        'min_sequence_interval': params['min_sequence_interval'] * random.uniform(0.9, 1.1),
                        'max_sequence_interval': params['max_sequence_interval'] * random.uniform(0.9, 1.1)
                    }
                    param_variations.append(variation)
            
            # Test all variations
            variation_results = []
            
            for var_params in param_variations:
                # Apply filters
                filtered_obs = filter_observations(
                    observations, 
                    include_nircam=var_params['include_nircam'],
                    max_ecliptic_latitude=var_params['max_ecliptic_latitude'], 
                    min_exposure_time=var_params['min_exposure_time']
                )
                
                # Group and find sequences
                fields = group_observations_by_field(filtered_obs)
                sequences = find_observation_sequences(
                    fields,
                    min_sequence_interval=var_params['min_sequence_interval'],
                    max_sequence_interval=var_params['max_sequence_interval']
                )
                
                # Calculate metrics
                filtered_percent = len(filtered_obs) / len(observations) * 100 if observations else 0
                fields_count = len(fields)
                sequences_count = len(sequences)
                
                # Quality metric (example)
                if sequences_count > 0:
                    avg_obs_per_seq = sum(seq['num_observations'] for seq in sequences) / sequences_count
                else:
                    avg_obs_per_seq = 0
                    
                quality_score = (sequences_count * avg_obs_per_seq) / (filtered_percent + 1)
                
                # Store results
                variation_results.append({
                    'parameters': var_params,
                    'filtered_count': len(filtered_obs),
                    'filtered_percent': filtered_percent,
                    'fields_count': fields_count,
                    'sequences_count': sequences_count,
                    'avg_observations_per_sequence': avg_obs_per_seq,
                    'quality_score': quality_score
                })
            
            # Aggregate variation results
            avg_quality = sum(r['quality_score'] for r in variation_results) / len(variation_results)
            avg_sequences = sum(r['sequences_count'] for r in variation_results) / len(variation_results)
            
            # Store aggregated results
            results.append({
                'parameter_set': params['name'],
                'average_quality_score': avg_quality,
                'average_sequences_count': avg_sequences,
                'variations': variation_results
            })
        
        # Sort by average quality score
        results.sort(key=lambda x: x['average_quality_score'], reverse=True)
        
        logger.info(f"Evaluated {len(test_params)} parameter sets")
        
        return {
            'parameter_tests': results,
            'best_parameter_set': results[0]['parameter_set'] if results else None
        }
    
    except Exception as e:
        logger.error(f"Error evaluating filter parameters: {e}")
        return {'error': str(e), 'parameter_tests': []}

def analyze_observation_quality(observations):
    """
    Analyze the quality of observations for KBO detection
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
        
    Returns:
    --------
    dict : Quality analysis results
    """
    if not observations:
        logger.info("No observations to analyze quality")
        return {'quality_metrics': {}}
    
    try:
        # Extract quality-related metrics
        exposure_times = []
        wavelengths = []
        ecliptic_distances = []
        
        for obs in observations:
            # Exposure time
            for field in ['t_exptime', 'exptime', 'exposure_time']:
                if field in obs:
                    try:
                        exposure_times.append(float(obs[field]))
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Wavelength
            if 'wavelength_range' in obs and obs['wavelength_range']:
                try:
                    wavelength = obs['wavelength_range']
                    if isinstance(wavelength, (list, tuple, np.ndarray)) and len(wavelength) >= 2:
                        wl_min, wl_max = wavelength[0], wavelength[1]
                        
                        # Convert to microns if needed
                        if wl_min < 1e-4 or wl_max < 1e-4:  # Likely in meters
                            wl_min *= 1e6
                            wl_max *= 1e6
                        
                        wavelengths.append((wl_min + wl_max) / 2)
                except (ValueError, TypeError):
                    pass
            
            # Ecliptic distance
            for ra_field, dec_field in [('s_ra', 's_dec'), ('ra', 'dec'), ('RA', 'DEC')]:
                if ra_field in obs and dec_field in obs:
                    try:
                        ra = float(obs[ra_field])
                        dec = float(obs[dec_field])
                        
                        # Convert to ecliptic coordinates
                        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
                        ecliptic_coord = coord.transform_to('ecliptic')
                        ecliptic_distances.append(abs(ecliptic_coord.lat.deg))
                        break
                    except (ValueError, TypeError):
                        pass
        
        # Calculate quality metrics
        quality_metrics = {}
        
        # Exposure time metrics
        if exposure_times:
            exposure_time_metrics = {
                'min': min(exposure_times),
                'max': max(exposure_times),
                'mean': np.mean(exposure_times),
                'median': np.median(exposure_times),
                'std': np.std(exposure_times),
                'optimal_percent': sum(1 for t in exposure_times if t >= 500) / len(exposure_times) * 100
            }
            quality_metrics['exposure_time'] = exposure_time_metrics
        
        # Wavelength metrics
        if wavelengths:
            wavelength_metrics = {
                'min': min(wavelengths),
                'max': max(wavelengths),
                'mean': np.mean(wavelengths),
                'median': np.median(wavelengths),
                'optimal_range': (10.0, 25.0),  # Ideal range for KBO detection
                'optimal_percent': sum(1 for w in wavelengths if 10.0 <= w <= 25.0) / len(wavelengths) * 100
            }
            quality_metrics['wavelength'] = wavelength_metrics
        
        # Ecliptic distance metrics
        if ecliptic_distances:
            ecliptic_metrics = {
                'min': min(ecliptic_distances),
                'max': max(ecliptic_distances),
                'mean': np.mean(ecliptic_distances),
                'median': np.median(ecliptic_distances),
                'optimal_percent': sum(1 for d in ecliptic_distances if d <= 5.0) / len(ecliptic_distances) * 100
            }
            quality_metrics['ecliptic_distance'] = ecliptic_metrics
        
        # Overall quality score (0-100)
        quality_score = 0
        components = 0
        
        if 'exposure_time' in quality_metrics:
            exp_score = min(100, quality_metrics['exposure_time']['optimal_percent'])
            quality_score += exp_score
            components += 1
            
        if 'wavelength' in quality_metrics:
            wl_score = min(100, quality_metrics['wavelength']['optimal_percent'])
            quality_score += wl_score
            components += 1
            
        if 'ecliptic_distance' in quality_metrics:
            ec_score = min(100, quality_metrics['ecliptic_distance']['optimal_percent'])
            quality_score += ec_score
            components += 1
        
        if components > 0:
            quality_score /= components
        
        # Quality rating
        if quality_score >= 80:
            quality_rating = "Excellent"
        elif quality_score >= 60:
            quality_rating = "Good"
        elif quality_score >= 40:
            quality_rating = "Fair"
        else:
            quality_rating = "Poor"
        
        quality_metrics['overall_score'] = quality_score
        quality_metrics['quality_rating'] = quality_rating
        
        logger.info(f"Analyzed quality of {len(observations)} observations")
        logger.info(f"Overall quality rating: {quality_rating} ({quality_score:.1f}%)")
        
        return {
            'quality_metrics': quality_metrics,
            'observation_count': len(observations)
        }
    
    except Exception as e:
        logger.error(f"Error analyzing observation quality: {e}")
        return {'error': str(e), 'quality_metrics': {}}

def generate_motion_models(sequences):
    """
    Generate models of possible KBO motions based on sequence data
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
        
    Returns:
    --------
    dict : Motion models for different KBO distances
    """
    if not sequences:
        logger.info("No sequences to generate motion models")
        return {'motion_models': {}}
    
    try:
        # Initialize models for different distance ranges
        models = {
            'nearby': {
                'distance_range': (20, 40),  # AU
                'motion_rate_range': (2.5, 5.0),  # arcsec/hour
                'sequences': []
            },
            'mid_range': {
                'distance_range': (40, 60),  # AU
                'motion_rate_range': (1.5, 2.5),  # arcsec/hour
                'sequences': []
            },
            'distant': {
                'distance_range': (60, 100),  # AU
                'motion_rate_range': (0.8, 1.5),  # arcsec/hour
                'sequences': []
            },
            'very_distant': {
                'distance_range': (100, 200),  # AU
                'motion_rate_range': (0.3, 0.8),  # arcsec/hour
                'sequences': []
            }
        }
        
        # Typical motion rate
        typical_rate = KBO_DETECTION_CONSTANTS['TYPICAL_MOTION_RATE']  # arcsec/hour
        
        # Assign sequences to models based on duration and quality
        for seq in sequences:
            duration = seq.get('duration_hours', 0)
            score = seq.get('kbo_score', 0)
            
            if duration <= 0:
                continue
                
            # Calculate expected motion for different distance ranges
            for model_name, model in models.items():
                min_rate, max_rate = model['motion_rate_range']
                avg_rate = (min_rate + max_rate) / 2
                
                expected_motion = avg_rate * duration
                
                # Add to model if sequence could detect this motion
                if expected_motion >= 2.0 and score >= 0.6:
                    model['sequences'].append({
                        'sequence': seq,
                        'expected_motion_arcsec': expected_motion
                    })
        
        # Calculate success probability for each model
        for model_name, model in models.items():
            if not model['sequences']:
                model['success_probability'] = 0.0
                continue
                
            # Weight by quality score
            total_score = sum(item['sequence'].get('kbo_score', 0) for item in model['sequences'])
            avg_score = total_score / len(model['sequences']) if model['sequences'] else 0
            
            # Weight by expected motion
            total_motion = sum(item['expected_motion_arcsec'] for item in model['sequences'])
            avg_motion = total_motion / len(model['sequences']) if model['sequences'] else 0
            
            # Combined probability
            # Higher score and motion are better
            prob = avg_score * min(1.0, avg_motion / 10.0)
            model['success_probability'] = prob
            
            # Calculate distance estimate
            min_dist, max_dist = model['distance_range']
            model['avg_distance_au'] = (min_dist + max_dist) / 2
            
            # Example orbit parameters for this distance
            avg_dist = model['avg_distance_au']
            model['orbital_period_years'] = np.power(avg_dist, 1.5)  # Kepler's third law
            model['orbital_velocity_km_s'] = 29.8 / np.sqrt(avg_dist)  # Approximation
        
        # Sort models by success probability
        sorted_models = {
            name: models[name] 
            for name in sorted(models, key=lambda x: models[x]['success_probability'], reverse=True)
        }
        
        logger.info(f"Generated motion models from {len(sequences)} sequences")
        
        return {
            'motion_models': sorted_models,
            'best_model': next(iter(sorted_models), None)
        }
    
    except Exception as e:
        logger.error(f"Error generating motion models: {e}")
        return {'error': str(e), 'motion_models': {}}

def plot_motion_model_results(motion_models, output_file=None):
    """
    Generate a visualization of motion model results
    
    Parameters:
    -----------
    motion_models : dict
        Dictionary of motion models from generate_motion_models
    output_file : str or None
        Output file path for the plot
        
    Returns:
    --------
    str or None : Path to output file if successful
    """
    if not motion_models or 'motion_models' not in motion_models:
        logger.warning("No motion models to visualize")
        return None
    
    try:
        models = motion_models['motion_models']
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Bar plot of success probabilities
        model_names = list(models.keys())
        probabilities = [models[name]['success_probability'] for name in model_names]
        
        # Format labels
        labels = []
        for name in model_names:
            model = models[name]
            dist_range = model['distance_range']
            label = f"{name}\n({dist_range[0]}-{dist_range[1]} AU)"
            labels.append(label)
        
        # Create bars
        bars = plt.bar(labels, probabilities, color='skyblue', alpha=0.8)
        
        # Add sequence counts
        for i, bar in enumerate(bars):
            name = model_names[i]
            seq_count = len(models[name]['sequences'])
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{seq_count} sequences",
                ha='center', va='bottom',
                fontsize=10
            )
        
        # Set labels and title
        plt.xlabel('KBO Distance Range')
        plt.ylabel('Detection Probability')
        plt.title('KBO Detection Probability by Distance Range')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add details table
        details = []
        for name in model_names:
            model = models[name]
            details.append([
                f"{name}",
                f"{model['avg_distance_au']:.1f} AU",
                f"{model['orbital_period_years']:.1f} yr",
                f"{len(model['sequences'])}"
            ])
        
        # Add a table below the chart
        plt.table(
            cellText=details,
            colLabels=['Model', 'Avg. Distance', 'Orbital Period', 'Sequences'],
            loc='bottom',
            bbox=[0.0, -0.35, 1.0, 0.2]
        )
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        
        # Save if output file specified
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved motion model visualization to {output_file}")
            plt.close()
            return output_file
        else:
            plt.show()
            plt.close()
            return None
    
    except Exception as e:
        logger.error(f"Error plotting motion model results: {e}")
        return None