"""
motion_calculator.py - Calculate expected KBO motion parameters

This module handles calculations of expected KBO motion rates, 
ranges, and vectors based on solar system physics.
"""

import numpy as np
import math

# Constants for KBO motion calculations
AU_TO_KM = 149597870.7  # km per AU
KM_PER_DEG_AT_40AU = 2618000.0  # km per degree at 40 AU
ARCSEC_PER_DEGREE = 3600.0

# KBO orbital speeds by distance (approximate values)
KBO_ORBITAL_SPEED_KM_PER_SEC = {
    30: 5.3,  # km/s at 30 AU (inner Kuiper Belt)
    40: 4.7,  # km/s at 40 AU (main Kuiper Belt)
    50: 4.2,  # km/s at 50 AU (outer Kuiper Belt)
    100: 3.0  # km/s at 100 AU (scattered disk objects)
}

def calculate_kbo_motion_range(time_interval_hours, plate_scale_arcsec_per_pixel=0.11, verbose=True):
    """
    Calculate the expected motion range for KBOs at different distances
    
    Parameters:
    -----------
    time_interval_hours : float
        Time interval between first and last image in hours
    plate_scale_arcsec_per_pixel : float
        Plate scale of the detector in arcseconds per pixel
    verbose : bool
        Whether to print verbose information
    
    Returns:
    --------
    dict : Dictionary of motion ranges in arcsec and pixels
    """
    if verbose:
        print("\nCalculating expected KBO motion range:")
        print(f"  Time interval: {time_interval_hours:.2f} hours")
        print(f"  Plate scale: {plate_scale_arcsec_per_pixel:.3f} arcsec/pixel")
    
    # Calculate motion in arcsec/hour at different distances
    motion_ranges = {}
    distances = [30, 40, 50, 100]  # AU
    
    for dist in distances:
        # Angular velocity (arcsec/hour) = orbital_speed (km/s) * 3600 (s/hour) / distance_per_arcsec (km/arcsec)
        distance_per_arcsec = KM_PER_DEG_AT_40AU * (dist / 40.0) / ARCSEC_PER_DEGREE
        angular_velocity = KBO_ORBITAL_SPEED_KM_PER_SEC[dist] * 3600.0 / distance_per_arcsec
        
        total_motion_arcsec = angular_velocity * time_interval_hours
        total_motion_pixels = total_motion_arcsec / plate_scale_arcsec_per_pixel
        
        motion_ranges[dist] = {
            'angular_velocity': angular_velocity,  # arcsec/hour
            'total_motion_arcsec': total_motion_arcsec,
            'total_motion_pixels': total_motion_pixels
        }
        
        if verbose:
            print(f"  At {dist} AU:")
            print(f"    Angular velocity: {angular_velocity:.2f} arcsec/hour")
            print(f"    Total motion: {total_motion_arcsec:.2f} arcsec = {total_motion_pixels:.2f} pixels")
    
    # Define min/max search range with buffer for uncertainty
    min_motion = motion_ranges[100]['total_motion_arcsec'] * 0.5  # 50% of motion at 100 AU
    max_motion = motion_ranges[30]['total_motion_arcsec'] * 1.5   # 150% of motion at 30 AU
    
    min_motion_pixels = min_motion / plate_scale_arcsec_per_pixel
    max_motion_pixels = max_motion / plate_scale_arcsec_per_pixel
    
    if verbose:
        print("\nSearch range:")
        print(f"  Min motion: {min_motion:.2f} arcsec = {min_motion_pixels:.2f} pixels")
        print(f"  Max motion: {max_motion:.2f} arcsec = {max_motion_pixels:.2f} pixels")
    
    return {
        'ranges': motion_ranges,
        'min_motion_arcsec': min_motion,
        'max_motion_arcsec': max_motion,
        'min_motion_pixels': min_motion_pixels,
        'max_motion_pixels': max_motion_pixels
    }

def generate_motion_vectors(min_speed, max_speed, num_steps=8, num_angles=16, verbose=True):
    """
    Generate a grid of motion vectors for KBO detection
    
    Parameters:
    -----------
    min_speed : float
        Minimum speed in pixels per image
    max_speed : float
        Maximum speed in pixels per image
    num_steps : int
        Number of speed steps to generate
    num_angles : int
        Number of angles to generate (full 360 degrees)
    verbose : bool
        Whether to print verbose information
    
    Returns:
    --------
    list : List of (dx, dy) motion vectors in pixels per image
    """
    if verbose:
        print("\nGenerating motion vectors:")
        print(f"  Speed range: {min_speed:.2f} to {max_speed:.2f} pixels per image")
        print(f"  Number of speeds: {num_steps}")
        print(f"  Number of angles: {num_angles}")
    
    motion_vectors = []
    
    # Use logarithmic spacing for speeds to handle large range
    if min_speed > 0 and max_speed > min_speed:
        speeds = np.geomspace(min_speed, max_speed, num_steps)
    else:
        speeds = np.linspace(min_speed, max_speed, num_steps)
    
    # Generate angles (full 360 degrees)
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    for speed in speeds:
        for angle in angles:
            dx = speed * np.cos(angle)
            dy = speed * np.sin(angle)
            motion_vectors.append((dx, dy))
    
    if verbose:
        print(f"  Generated {len(motion_vectors)} motion vectors")
        if len(motion_vectors) > 0:
            print("  Vector examples:")
            for i, (dx, dy) in enumerate(motion_vectors[:min(5, len(motion_vectors))]):
                print(f"    Vector {i+1}: dx={dx:.3f}, dy={dy:.3f} pixels per image")
            if len(motion_vectors) > 5:
                print(f"    ... and {len(motion_vectors)-5} more")
    
    return motion_vectors

def calculate_shifts_for_motion(motion_vector, time_intervals):
    """
    Calculate shifts for each image based on a motion vector and time intervals
    
    Parameters:
    -----------
    motion_vector : tuple
        (dx, dy) motion vector in pixels for the entire sequence
    time_intervals : list
        List of time intervals from the first image (in hours)
        
    Returns:
    --------
    list : List of (dx, dy) shifts for each image
    """
    dx_total, dy_total = motion_vector
    
    # First image (reference) has no shift
    shifts = [(0, 0)]
    
    # Calculate time fraction and corresponding shift for each interval
    total_time = time_intervals[-1]
    
    for interval in time_intervals:
        fraction = interval / total_time
        dx = dx_total * fraction
        dy = dy_total * fraction
        shifts.append((dx, dy))
    
    return shifts

def filter_motion_vectors(motion_vectors, plate_scale_arcsec_per_pixel, 
                         time_span_hours, max_vectors=None):
    """
    Filter motion vectors based on physical constraints
    
    Parameters:
    -----------
    motion_vectors : list
        List of (dx, dy) motion vectors
    plate_scale_arcsec_per_pixel : float
        Plate scale in arcseconds per pixel
    time_span_hours : float
        Time span of the observation in hours
    max_vectors : int or None
        Maximum number of vectors to return
        
    Returns:
    --------
    list : Filtered list of motion vectors
    """
    if not motion_vectors:
        return []
        
    # Constants from KBO physics
    MIN_ARCSEC_PER_HOUR = 0.5  # Minimum plausible KBO motion
    MAX_ARCSEC_PER_HOUR = 10.0  # Maximum plausible KBO motion
    
    # Filter vectors based on physical constraints
    filtered_vectors = []
    
    for dx, dy in motion_vectors:
        # Calculate total motion in pixels
        speed_pixels = math.sqrt(dx*dx + dy*dy)
        
        # Convert to arcsec/hour
        speed_arcsec = speed_pixels * plate_scale_arcsec_per_pixel
        speed_arcsec_per_hour = speed_arcsec / time_span_hours if time_span_hours > 0 else 0
        
        # Check if within physical limits for KBOs
        if (MIN_ARCSEC_PER_HOUR <= speed_arcsec_per_hour <= MAX_ARCSEC_PER_HOUR):
            filtered_vectors.append((dx, dy))
    
    # If a maximum number of vectors is specified, sample evenly
    if max_vectors is not None and len(filtered_vectors) > max_vectors:
        # Sort by speed to ensure we get a good distribution
        filtered_vectors.sort(key=lambda v: v[0]**2 + v[1]**2)
        
        # Select vectors evenly spaced from the sorted list
        step = len(filtered_vectors) / max_vectors
        indices = [int(i * step) for i in range(max_vectors)]
        filtered_vectors = [filtered_vectors[i] for i in indices]
    
    return filtered_vectors