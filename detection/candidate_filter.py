"""
candidate_filter.py - Filter and score KBO candidates

This module handles filtering, scoring, and validation of potential KBO candidates
based on physical constraints and observation parameters.
"""

import numpy as np
import math

# Physical constraints for KBOs
KBO_CONSTRAINTS = {
    # Motion constraints
    'min_arcsec_per_hour': 0.5,  # Minimum motion in arcsec/hour (too slow = stationary object)
    'max_arcsec_per_hour': 10.0,  # Maximum motion in arcsec/hour (too fast = asteroid/error)
    
    # Object property constraints
    'min_snr': 3.0,            # Minimum signal-to-noise ratio
    'max_fwhm': 5.0,           # Maximum full width at half maximum (too large = extended source)
    'min_detections': 2,       # Minimum number of frames where object should be detectable
    
    # Solar system physics constraints
    'min_distance_au': 30.0,   # Minimum distance in AU (inner Kuiper Belt)
    'max_distance_au': 100.0,  # Maximum distance in AU (outer detectable KBOs)
    
    # Score thresholds
    'min_score': 0.7,          # Minimum score to keep a candidate
    'high_score': 0.8          # Score considered highly reliable
}

def filter_candidates(candidates, time_span_hours, plate_scale_arcsec_per_pixel, verbose=True):
    """
    Filter candidates based on physical and observation constraints
    
    Parameters:
    -----------
    candidates : list
        List of KBO candidates
    time_span_hours : float
        Time span of the observation in hours
    plate_scale_arcsec_per_pixel : float
        Plate scale in arcseconds per pixel
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    list
        Filtered list of candidates
    """
    if not candidates:
        return []
        
    if verbose:
        print("\nFiltering candidates based on physical constraints...")
        print(f"Starting with {len(candidates)} candidates")
    
    filtered = []
    rejected_reasons = {
        'low_score': 0,
        'too_fast': 0,
        'too_slow': 0,
        'nonlinear': 0,
        'edge_object': 0,
        'low_snr': 0,
        'extended': 0,
        'other': 0
    }
    
    for candidate in candidates:
        # Extract motion parameters
        dx, dy = candidate['motion_vector']
        motion_pixels = math.sqrt(dx*dx + dy*dy) * len(candidate.get('shifts', []))
        motion_arcsec = motion_pixels * plate_scale_arcsec_per_pixel
        motion_arcsec_per_hour = motion_arcsec / time_span_hours if time_span_hours > 0 else 0
        
        # Check score
        if candidate['score'] < KBO_CONSTRAINTS['min_score']:
            rejected_reasons['low_score'] += 1
            continue
            
        # Check motion rate
        if motion_arcsec_per_hour < KBO_CONSTRAINTS['min_arcsec_per_hour']:
            rejected_reasons['too_slow'] += 1
            continue
            
        if motion_arcsec_per_hour > KBO_CONSTRAINTS['max_arcsec_per_hour']:
            rejected_reasons['too_fast'] += 1
            continue
            
        # Calculate approximate distance based on motion rate
        # For a circular orbit: distance_au ≈ 4.74 / motion_arcsec_per_hour
        # (simplified formula based on orbital mechanics)
        approx_distance_au = 4.74 / motion_arcsec_per_hour
        
        # Check if in reasonable KBO distance range
        if approx_distance_au < KBO_CONSTRAINTS['min_distance_au'] or \
           approx_distance_au > KBO_CONSTRAINTS['max_distance_au']:
            rejected_reasons['other'] += 1
            continue
            
        # Add the filtered candidate with additional information
        candidate['motion_arcsec_per_hour'] = motion_arcsec_per_hour
        candidate['approx_distance_au'] = approx_distance_au
        filtered.append(candidate)
    
    if verbose:
        print(f"After filtering: {len(filtered)}/{len(candidates)} candidates passed")
        print("Rejection reasons:")
        for reason, count in rejected_reasons.items():
            if count > 0:
                print(f"  {reason}: {count}")
    
    # Sort by score (descending)
    filtered.sort(key=lambda c: c['score'], reverse=True)
    
    return filtered

def score_candidate(candidate, images, original_shifts, search_radius=3):
    """
    Score a KBO candidate based on multiple criteria
    
    Parameters:
    -----------
    candidate : dict
        Candidate object information
    images : list
        List of original image arrays
    original_shifts : list
        List of shifts for each image
    search_radius : int
        Radius to search around expected position
    
    Returns:
    --------
    float
        Score between 0 and 1
    """
    # Extract candidate properties
    x, y = candidate['xcentroid'], candidate['ycentroid']
    motion_vector = candidate['motion_vector']
    
    # Base score starts at 0.5
    score = 0.5
    
    # Check if candidate is near the edge of any image (penalty: -0.2)
    for image in images:
        if (x < 10 or x > image.shape[1] - 10 or
            y < 10 or y > image.shape[0] - 10):
            score -= 0.2
            break
    
    # Penalize extremely large motion vectors (penalty: up to -0.4)
    dx, dy = motion_vector
    total_motion = math.sqrt(dx*dx + dy*dy) * len(images)
    if total_motion > 1000:  # Extremely large motions are suspicious
        motion_penalty = min(0.4, (total_motion - 1000) / 10000 * 0.4)
        score -= motion_penalty
    
    # Check consistency of candidate across images (bonus: up to +0.5)
    detections = 0
    expected_detections = 0
    snr_values = []
    
    for i, image in enumerate(images):
        # Skip reference image (first image)
        if i == 0:
            continue
            
        expected_detections += 1
        
        # Calculate expected position in this image
        shift_x, shift_y = original_shifts[i]
        dx = i * motion_vector[0]
        dy = i * motion_vector[1]
        
        expected_x = x - dx - shift_x
        expected_y = y - dy - shift_y
        
        # Skip if outside image bounds
        if (expected_x < 0 or expected_x >= image.shape[1] or
            expected_y < 0 or expected_y >= image.shape[0]):
            continue
        
        # Extract region around expected position
        x_min = max(0, int(expected_x - search_radius))
        x_max = min(image.shape[1], int(expected_x + search_radius + 1))
        y_min = max(0, int(expected_y - search_radius))
        y_max = min(image.shape[0], int(expected_y + search_radius + 1))
        
        region = image[y_min:y_max, x_min:x_max]
        
        # Skip if all NaN
        if np.all(np.isnan(region)):
            continue
        
        # Calculate local statistics
        local_median = np.nanmedian(region)
        local_std = np.nanstd(region)
        
        # Look for peak in the region
        peak_value = np.nanmax(region)
        snr = (peak_value - local_median) / local_std if local_std > 0 else 0
        
        if snr > 2.0:
            detections += 1
            snr_values.append(snr)
    
    # Calculate detection rate and adjust score (0-0.5 bonus)
    if expected_detections > 0:
        detection_rate = detections / expected_detections
        detection_bonus = 0.5 * detection_rate
        score += detection_bonus
        
        # Additional bonus for high SNR detections (up to +0.2)
        if snr_values:
            avg_snr = sum(snr_values) / len(snr_values)
            snr_bonus = min(0.2, (avg_snr - 2.0) / 10.0 * 0.2)
            score += snr_bonus
    
    # Ensure score is between 0 and 1
    score = max(0.0, min(1.0, score))
    
    return score

def calculate_kbo_properties(candidate, plate_scale_arcsec_per_pixel, time_span_hours):
    """
    Calculate physical properties of a KBO candidate
    
    Parameters:
    -----------
    candidate : dict
        Candidate object information
    plate_scale_arcsec_per_pixel : float
        Plate scale in arcseconds per pixel
    time_span_hours : float
        Time span of the observation in hours
        
    Returns:
    --------
    dict
        Physical properties of the KBO
    """
    # Extract motion parameters
    dx, dy = candidate['motion_vector']
    motion_pixels = math.sqrt(dx*dx + dy*dy) * len(candidate.get('shifts', []))
    motion_arcsec = motion_pixels * plate_scale_arcsec_per_pixel
    motion_arcsec_per_hour = motion_arcsec / time_span_hours if time_span_hours > 0 else 0
    
    # Calculate approximate distance based on motion rate
    # This is a simplification based on orbital mechanics for objects near the ecliptic
    approx_distance_au = 4.74 / motion_arcsec_per_hour if motion_arcsec_per_hour > 0 else 0
    
    # Calculate approximate size (very rough estimate)
    # This would require photometric calibration for a real estimate
    approx_size_km = "Unknown (photometric calibration needed)"
    
    # Calculate orbital properties
    motion_angle_deg = math.degrees(math.atan2(dy, dx))
    
    return {
        'motion_pixels_total': motion_pixels,
        'motion_arcsec_total': motion_arcsec,
        'motion_arcsec_per_hour': motion_arcsec_per_hour,
        'motion_angle_deg': motion_angle_deg,
        'approx_distance_au': approx_distance_au,
        'approx_size': approx_size_km
    }