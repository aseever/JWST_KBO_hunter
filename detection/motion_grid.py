"""
motion_grid.py - Generate motion vectors for KBO detection

This module calculates the expected motion of KBOs based on orbital dynamics
and generates grids of motion vectors to test in the shift-and-stack algorithm.
"""

import numpy as np
import math

# Constants for KBO motion calculations
AU_TO_KM = 149597870.7  # km per AU
KM_PER_DEG_AT_40AU = 2618000.0  # km per degree at 40 AU
KBO_ORBITAL_SPEED_KM_PER_SEC = {
    30: 5.3,  # km/s at 30 AU
    40: 4.7,  # km/s at 40 AU
    50: 4.2,  # km/s at 50 AU
    100: 3.0  # km/s at 100 AU
}
ARCSEC_PER_DEGREE = 3600.0

def calculate_kbo_motion_range(time_interval_hours, plate_scale_arcsec_per_pixel=0.11, verbose=True):
    """
    Calculate the expected motion range for KBOs at different distances
    
    Parameters:
    -----------
    time_interval_hours : float
        Time interval between first and last image in hours
    plate_scale_arcsec_per_pixel : float
        Plate scale of the detector in arcseconds per pixel (default: 0.11 for MIRI)
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
    
    # Define min/max search range (include buffer for uncertainty)
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

def generate_linear_motion_grid(min_speed, max_speed, num_speeds=8, num_angles=16, verbose=True):
    """
    Generate a grid of linear motion vectors to test
    
    Parameters:
    -----------
    min_speed : float
        Minimum speed in pixels per image
    max_speed : float
        Maximum speed in pixels per image
    num_speeds : int
        Number of speeds to test between min and max
    num_angles : int
        Number of angles to test (full 360 degrees)
    verbose : bool
        Whether to print verbose information
    
    Returns:
    --------
    list : List of (dx, dy) motion vectors in pixels per image
    """
    if verbose:
        print("\nGenerating motion vectors:")
        print(f"  Speed range: {min_speed:.2f} to {max_speed:.2f} pixels")
        print(f"  Number of speeds: {num_speeds}")
        print(f"  Number of angles: {num_angles}")
    
    motion_vectors = []
    
    # Generate speeds
    speeds = np.linspace(min_speed, max_speed, num_speeds)
    
    # Generate angles (full 360 degrees)
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    for speed in speeds:
        for angle in angles:
            dx = speed * np.cos(angle)
            dy = speed * np.sin(angle)
            motion_vectors.append((dx, dy))
    
    if verbose:
        print(f"  Generated {len(motion_vectors)} motion vectors")
        print(f"  Example vectors:")
        for i, (dx, dy) in enumerate(motion_vectors[:5]):
            print(f"    Vector {i+1}: dx={dx:.2f}, dy={dy:.2f} pixels")
        if len(motion_vectors) > 5:
            print(f"    ... and {len(motion_vectors)-5} more")
    
    return motion_vectors

def generate_motion_vectors_for_sequence(timestamps, plate_scale=0.11, factor=1.0, verbose=True):
    """
    Generate motion vectors based on the actual timestamps of a sequence
    
    Parameters:
    -----------
    timestamps : list
        List of observation timestamps (MJD)
    plate_scale : float
        Plate scale in arcseconds per pixel
    factor : float
        Scaling factor for motion ranges (use >1 for wider search)
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    dict : Contains motion ranges and per-image motion vectors
    """
    if len(timestamps) < 2:
        raise ValueError("Need at least two timestamps to calculate motion")
    
    # Sort timestamps
    sorted_timestamps = sorted(timestamps)
    
    # Calculate time span in hours
    time_span_days = sorted_timestamps[-1] - sorted_timestamps[0]
    time_span_hours = time_span_days * 24.0
    
    # Calculate intervals between images
    intervals = []
    for i in range(1, len(sorted_timestamps)):
        interval = (sorted_timestamps[i] - sorted_timestamps[0]) * 24.0  # hours
        intervals.append(interval)
    
    if verbose:
        print(f"Time span: {time_span_hours:.2f} hours")
        print(f"Time intervals from first image:")
        for i, interval in enumerate(intervals):
            print(f"  Image {i+2}: {interval:.2f} hours")
    
    # Calculate motion ranges
    motion_range = calculate_kbo_motion_range(time_span_hours, plate_scale, verbose)
    
    # Scale motion ranges by factor
    min_pixels = motion_range['min_motion_pixels'] * factor
    max_pixels = motion_range['max_motion_pixels'] * factor
    
    # Calculate per-image motion based on the longest time span
    speeds_per_image = []
    for i in range(len(timestamps)):
        if i == 0:
            # First image has no motion (reference)
            speeds_per_image.append({
                'min': 0.0,
                'max': 0.0
            })
        else:
            # Calculate fraction of total time span
            fraction = intervals[i-1] / time_span_hours
            speeds_per_image.append({
                'min': min_pixels * fraction,
                'max': max_pixels * fraction
            })
    
    # Generate motion vectors for the full time span
    num_speeds = max(8, min(32, int(max_pixels - min_pixels)))
    num_angles = max(16, min(64, num_speeds * 2))
    
    motion_vectors = generate_linear_motion_grid(min_pixels, max_pixels, num_speeds, num_angles, verbose)
    
    return {
        'time_span_hours': time_span_hours,
        'intervals': intervals,
        'motion_range': motion_range,
        'speeds_per_image': speeds_per_image,
        'motion_vectors': motion_vectors
    }

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

def filter_motion_vectors(motion_vectors, max_vectors=100):
    """
    Filter motion vectors to reduce computational load
    
    Parameters:
    -----------
    motion_vectors : list
        List of (dx, dy) motion vectors
    max_vectors : int
        Maximum number of vectors to return
        
    Returns:
    --------
    list : Filtered list of motion vectors
    """
    if len(motion_vectors) <= max_vectors:
        return motion_vectors
    
    # Calculate the step size to get close to max_vectors
    step = max(1, len(motion_vectors) // max_vectors)
    
    return motion_vectors[::step]
def generate_motion_vectors(min_speed, max_speed, num_steps=8, num_angles=16, verbose=True):
    """
    Generate a grid of motion vectors to test for KBO detection
    
    Parameters:
    -----------
    min_speed : float
        Minimum speed in pixels per image
    max_speed : float
        Maximum speed in pixels per image
    num_steps : int
        Number of speed steps between min and max
    num_angles : int
        Number of angles to test (evenly distributed over 360 degrees)
    verbose : bool
        Whether to print verbose information
    
    Returns:
    --------
    list : List of (dx, dy) motion vectors in pixels per image
    """
    # This is essentially an alias for generate_linear_motion_grid
    return generate_linear_motion_grid(min_speed, max_speed, num_steps, num_angles, verbose)