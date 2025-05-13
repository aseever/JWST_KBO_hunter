"""
mast/filter/sequences.py - Sequence detection and analysis for KBO observations

This module handles grouping observations by field, detecting time sequences,
calculating motion parameters, and scoring sequences for KBO detection potential.
"""

import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

# Set up logger
logger = logging.getLogger('mast_kbo')

# Import utilities
try:
    from mast.utils import KBO_DETECTION_CONSTANTS
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
    if not observations:
        logger.warning("No observations to group by field")
        return {}
        
    try:
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
    
    except Exception as e:
        logger.error(f"Error grouping observations by field: {e}")
        return {'error': [obs for obs in observations]}  # Return all observations in an error group

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
    if not fields:
        logger.warning("No fields to find sequences in")
        return []
        
    try:
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
            
            # Get field center coordinates if available
            field_center_ra = None
            field_center_dec = None
            
            # Look for coordinate fields in first observation
            coord_fields = [('s_ra', 's_dec'), ('ra', 'dec'), ('RA', 'DEC')]
            
            if observations:
                first_obs = observations[0]
                for ra_f, dec_f in coord_fields:
                    if ra_f in first_obs and dec_f in first_obs:
                        try:
                            field_center_ra = float(first_obs[ra_f])
                            field_center_dec = float(first_obs[dec_f])
                            break
                        except (ValueError, TypeError):
                            pass
            
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
                            'center_ra': field_center_ra,
                            'center_dec': field_center_dec,
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
                    'center_ra': field_center_ra,
                    'center_dec': field_center_dec,
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
    
    except Exception as e:
        logger.error(f"Error finding observation sequences: {e}")
        return []

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
    if not sequences:
        return []
        
    try:
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
                    
                    # Add orbital period estimate (in years)
                    # P ≈ a^(3/2) where a is distance in AU
                    sequence['approx_orbital_period_years'] = np.power(approx_distance_au, 1.5)
        
        return sequences
    
    except Exception as e:
        logger.error(f"Error calculating KBO motion parameters: {e}")
        return sequences  # Return sequences without added parameters

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
    if not sequences:
        return []
        
    try:
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
            if 'center_dec' in sequence and sequence['center_dec'] is not None:
                if abs(sequence.get('center_dec', 90)) < 5.0:
                    score += 0.1
            
            # Bonus for expected motion (if calculated)
            if 'expected_motion_arcsec' in sequence:
                motion = sequence['expected_motion_arcsec']
                # Higher score for motions between 10-60 arcsec
                if 10 <= motion <= 60:
                    score += 0.1
            
            # Ensure score is between 0 and 1
            sequence['kbo_score'] = max(0.0, min(1.0, score))
        
        # Sort by score
        sequences.sort(key=lambda x: x['kbo_score'], reverse=True)
        
        return sequences
    
    except Exception as e:
        logger.error(f"Error scoring sequences: {e}")
        return sequences  # Return sequences without scores

def compute_sequence_statistics(sequences):
    """
    Compute statistics about the sequence dataset
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
        
    Returns:
    --------
    dict : Dictionary with sequence statistics
    """
    if not sequences:
        return {
            'count': 0,
            'avg_observations': 0,
            'avg_duration': 0,
            'avg_score': 0,
            'high_quality_count': 0
        }
    
    try:
        # Basic statistics
        sequence_count = len(sequences)
        num_observations = [seq['num_observations'] for seq in sequences]
        durations = [seq.get('duration_hours', 0) for seq in sequences]
        scores = [seq.get('kbo_score', 0) for seq in sequences]
        
        # Filter high-quality sequences (score > 0.7)
        high_quality = [seq for seq in sequences if seq.get('kbo_score', 0) > 0.7]
        
        stats = {
            'count': sequence_count,
            'avg_observations': np.mean(num_observations) if num_observations else 0,
            'min_observations': min(num_observations) if num_observations else 0,
            'max_observations': max(num_observations) if num_observations else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'avg_score': np.mean(scores) if scores else 0,
            'high_quality_count': len(high_quality),
            'high_quality_percent': (len(high_quality) / sequence_count * 100) if sequence_count else 0
        }
        
        # Distribution by observation count
        obs_distribution = defaultdict(int)
        for n in num_observations:
            obs_distribution[n] += 1
        
        stats['observations_distribution'] = dict(obs_distribution)
        
        # Distribution by score ranges
        score_ranges = {
            'low': len([s for s in scores if s < 0.4]),
            'medium': len([s for s in scores if 0.4 <= s < 0.7]),
            'high': len([s for s in scores if s >= 0.7])
        }
        
        stats['score_distribution'] = score_ranges
        
        return stats
    
    except Exception as e:
        logger.error(f"Error computing sequence statistics: {e}")
        return {'count': len(sequences), 'error': str(e)}

def find_similar_sequences(sequence, all_sequences, time_threshold_hours=48.0, spatial_threshold_deg=1.0):
    """
    Find sequences similar to a given sequence
    
    Parameters:
    -----------
    sequence : dict
        The reference sequence
    all_sequences : list
        List of all sequences to compare against
    time_threshold_hours : float
        Maximum time difference to consider similar
    spatial_threshold_deg : float
        Maximum spatial separation to consider similar
        
    Returns:
    --------
    list : List of similar sequences
    """
    if not sequence or not all_sequences:
        return []
    
    try:
        # Extract reference sequence properties
        ref_start = Time(sequence.get('start_time', '2000-01-01'), format='isot').mjd
        ref_end = Time(sequence.get('end_time', '2000-01-01'), format='isot').mjd
        ref_ra = sequence.get('center_ra')
        ref_dec = sequence.get('center_dec')
        
        similar_sequences = []
        
        for seq in all_sequences:
            # Skip if it's the same sequence
            if seq is sequence:
                continue
                
            try:
                # Check time proximity
                seq_start = Time(seq.get('start_time', '2000-01-01'), format='isot').mjd
                seq_end = Time(seq.get('end_time', '2000-01-01'), format='isot').mjd
                
                time_diff_hours_start = abs(seq_start - ref_start) * 24.0
                time_diff_hours_end = abs(seq_end - ref_end) * 24.0
                
                # Check spatial proximity
                seq_ra = seq.get('center_ra')
                seq_dec = seq.get('center_dec')
                
                if (ref_ra is not None and ref_dec is not None and 
                    seq_ra is not None and seq_dec is not None):
                    
                    ref_coord = SkyCoord(ra=ref_ra, dec=ref_dec, unit=(u.deg, u.deg))
                    seq_coord = SkyCoord(ra=seq_ra, dec=seq_dec, unit=(u.deg, u.deg))
                    
                    # Calculate angular separation
                    separation = ref_coord.separation(seq_coord).deg
                    
                    # If both time and space are within threshold, it's similar
                    if (time_diff_hours_start <= time_threshold_hours or 
                        time_diff_hours_end <= time_threshold_hours) and \
                       separation <= spatial_threshold_deg:
                        
                        similarity = {
                            'sequence': seq,
                            'time_diff_hours': min(time_diff_hours_start, time_diff_hours_end),
                            'spatial_separation_deg': separation
                        }
                        similar_sequences.append(similarity)
            except Exception:
                # Skip sequences with missing or invalid data
                continue
        
        # Sort by spatial separation
        similar_sequences.sort(key=lambda x: x['spatial_separation_deg'])
        
        return similar_sequences
    
    except Exception as e:
        logger.error(f"Error finding similar sequences: {e}")
        return []

def merge_sequences(sequences, max_gap_hours=48.0, max_spatial_separation_deg=0.2):
    """
    Merge sequences that likely observe the same object
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
    max_gap_hours : float
        Maximum time gap between sequences to merge
    max_spatial_separation_deg : float
        Maximum spatial separation to merge sequences
        
    Returns:
    --------
    list : List of merged sequences
    """
    if not sequences:
        return []
    
    try:
        # Make a copy of the sequences to avoid modifying the originals
        working_sequences = sequences.copy()
        merged_sequences = []
        processed = set()
        
        for i, seq1 in enumerate(sequences):
            if i in processed:
                continue
                
            # Start a new merged sequence with this one
            current_merged = {
                'field_id': seq1['field_id'],
                'center_ra': seq1.get('center_ra'),
                'center_dec': seq1.get('center_dec'),
                'num_observations': seq1['num_observations'],
                'start_time': seq1.get('start_time'),
                'end_time': seq1.get('end_time'),
                'duration_hours': seq1.get('duration_hours', 0),
                'observations': seq1.get('observations', []),
                'merged_from': [i],
                'kbo_score': seq1.get('kbo_score', 0)
            }
            
            # Mark as processed
            processed.add(i)
            
            # Look for sequences to merge
            merged_something = True
            while merged_something:
                merged_something = False
                
                for j, seq2 in enumerate(sequences):
                    if j in processed:
                        continue
                        
                    # Check if sequences can be merged
                    try:
                        # Check spatial proximity
                        if (current_merged['center_ra'] is not None and 
                            current_merged['center_dec'] is not None and 
                            seq2.get('center_ra') is not None and 
                            seq2.get('center_dec') is not None):
                            
                            coord1 = SkyCoord(
                                ra=current_merged['center_ra'], 
                                dec=current_merged['center_dec'], 
                                unit=(u.deg, u.deg)
                            )
                            
                            coord2 = SkyCoord(
                                ra=seq2.get('center_ra'), 
                                dec=seq2.get('center_dec'), 
                                unit=(u.deg, u.deg)
                            )
                            
                            separation = coord1.separation(coord2).deg
                            
                            # Check time proximity
                            time1_end = Time(current_merged['end_time'], format='isot').mjd
                            time2_start = Time(seq2.get('start_time', '2000-01-01'), format='isot').mjd
                            
                            time_gap = (time2_start - time1_end) * 24.0  # Convert to hours
                            
                            # If within thresholds, merge
                            if separation <= max_spatial_separation_deg and time_gap <= max_gap_hours:
                                # Update merged sequence
                                current_merged['end_time'] = seq2.get('end_time')
                                current_merged['num_observations'] += seq2['num_observations']
                                
                                # Recalculate duration
                                start_time = Time(current_merged['start_time'], format='isot').mjd
                                end_time = Time(current_merged['end_time'], format='isot').mjd
                                current_merged['duration_hours'] = (end_time - start_time) * 24.0
                                
                                # Merge observations
                                current_merged['observations'].extend(seq2.get('observations', []))
                                
                                # Update merged_from
                                current_merged['merged_from'].append(j)
                                
                                # Update score (use max of the two)
                                current_merged['kbo_score'] = max(
                                    current_merged['kbo_score'],
                                    seq2.get('kbo_score', 0)
                                )
                                
                                # Mark as processed
                                processed.add(j)
                                
                                # Continue searching for more to merge
                                merged_something = True
                    except Exception:
                        # Skip if there's an error comparing
                        continue
            
            # Add the merged sequence
            merged_sequences.append(current_merged)
        
        # Log merge results
        if len(merged_sequences) < len(sequences):
            logger.info(f"Merged {len(sequences)} sequences into {len(merged_sequences)} sequences")
        else:
            logger.info("No sequences were merged")
        
        # Sort by score
        merged_sequences.sort(key=lambda x: x.get('kbo_score', 0), reverse=True)
        
        return merged_sequences
    
    except Exception as e:
        logger.error(f"Error merging sequences: {e}")
        return sequences  # Return original sequences on error