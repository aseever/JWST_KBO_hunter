"""
shift_stack.py - Core implementation of shift-and-stack algorithm for KBO detection

This module provides functions to shift and stack images according to hypothetical
motion vectors, which allows detection of moving objects that would be too faint
to detect in single frames.
"""

import numpy as np
from scipy import ndimage
import warnings

# Don't import from source_finder here to avoid circular imports
# Instead, we'll import it inside the process_motion_vector function

def apply_shift(image, dx, dy, order=1, mode='constant', cval=np.nan):
    """
    Shift an image by a given displacement
    
    Parameters:
    -----------
    image : numpy.ndarray
        2D image data
    dx, dy : float
        Displacement in x and y directions (pixels)
    order : int
        Order of spline interpolation (0-5)
    mode : str
        How to handle points outside the boundaries
    cval : float
        Value to fill outside boundary when mode='constant'
    
    Returns:
    --------
    numpy.ndarray : Shifted image
    """
    # Use scipy.ndimage.shift for subpixel shifts
    return ndimage.shift(image, (dy, dx), order=order, mode=mode, cval=cval)

def stack_images(images, shifts, method='mean', weights=None, mask_nans=True):
    """
    Stack images after applying shifts
    
    Parameters:
    -----------
    images : list
        List of 2D image arrays
    shifts : list
        List of (dx, dy) shifts for each image
    method : str
        Stacking method: 'mean', 'median', 'sum', or 'weighted_mean'
    weights : list or None
        List of weights for weighted mean (requires method='weighted_mean')
    mask_nans : bool
        Whether to mask NaN values when stacking
    
    Returns:
    --------
    numpy.ndarray : Stacked image
    """
    if len(images) != len(shifts):
        raise ValueError("Number of images and shifts must match")
    
    if method == 'weighted_mean' and (weights is None or len(weights) != len(images)):
        raise ValueError("Weighted mean requires weights for each image")
    
    # Apply shifts to each image
    shifted_images = []
    for i, image in enumerate(images):
        dx, dy = shifts[i]
        shifted = apply_shift(image, dx, dy)
        shifted_images.append(shifted)
    
    # Stack the shifted images
    if mask_nans:
        # Use nanmean/nanmedian/nansum to ignore NaN values
        if method == 'mean':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stacked = np.nanmean(shifted_images, axis=0)
        elif method == 'median':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stacked = np.nanmedian(shifted_images, axis=0)
        elif method == 'sum':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stacked = np.nansum(shifted_images, axis=0)
        elif method == 'weighted_mean':
            # Implement weighted mean with NaN handling
            stacked = np.zeros_like(shifted_images[0])
            weight_sum = np.zeros_like(shifted_images[0])
            
            for i, img in enumerate(shifted_images):
                mask = ~np.isnan(img)
                stacked[mask] += img[mask] * weights[i]
                weight_sum[mask] += weights[i]
            
            # Normalize by sum of weights, avoiding division by zero
            mask = weight_sum > 0
            stacked[mask] /= weight_sum[mask]
            stacked[~mask] = np.nan
        else:
            raise ValueError(f"Unknown stacking method: {method}")
    else:
        # Standard NumPy functions (will propagate NaNs)
        if method == 'mean':
            stacked = np.mean(shifted_images, axis=0)
        elif method == 'median':
            stacked = np.median(shifted_images, axis=0)
        elif method == 'sum':
            stacked = np.sum(shifted_images, axis=0)
        elif method == 'weighted_mean':
            stacked = np.average(shifted_images, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown stacking method: {method}")
    
    return stacked

def shift_and_stack(images, motion_vector, time_intervals, method='mean', weights=None):
    """
    Apply shift-and-stack for a single motion vector
    
    Parameters:
    -----------
    images : list
        List of 2D image arrays
    motion_vector : tuple
        (dx, dy) motion vector in pixels for the entire sequence
    time_intervals : list
        List of time intervals from the first image in hours 
        (should be one element shorter than images list)
    method : str
        Stacking method: 'mean', 'median', 'sum', or 'weighted_mean'
    weights : list or None
        List of weights for weighted mean
    
    Returns:
    --------
    tuple : (stacked_image, shifts)
    """
    # Validate inputs
    if len(images) != len(time_intervals) + 1:
        raise ValueError("time_intervals should have one less element than images")
    
    # Calculate shifts for each image
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
    
    # Stack the images with these shifts
    stacked = stack_images(images, shifts, method=method, weights=weights)
    
    return stacked, shifts

def multi_shift_stack(images, motion_vectors, time_intervals, method='mean', max_vectors=None):
    """
    Apply shift-and-stack for multiple motion vectors
    
    Parameters:
    -----------
    images : list
        List of 2D image arrays
    motion_vectors : list
        List of (dx, dy) motion vectors
    time_intervals : list
        List of time intervals from the first image
    method : str
        Stacking method
    max_vectors : int or None
        Maximum number of motion vectors to process (for limiting computation)
    
    Returns:
    --------
    list : List of (stacked_image, motion_vector, shifts) tuples
    """
    # Limit number of vectors if requested
    if max_vectors is not None and len(motion_vectors) > max_vectors:
        # Take a subset of vectors
        step = len(motion_vectors) // max_vectors
        vectors_to_use = motion_vectors[::step][:max_vectors]
    else:
        vectors_to_use = motion_vectors
    
    results = []
    
    for motion_vector in vectors_to_use:
        stacked, shifts = shift_and_stack(images, motion_vector, time_intervals, method)
        results.append((stacked, motion_vector, shifts))
    
    return results

def shift_stack_pixel_grid(images, time_intervals, max_shift=10, step=1, method='mean'):
    """
    Apply shift-and-stack on a regular grid of integer pixel shifts
    
    Parameters:
    -----------
    images : list
        List of 2D image arrays
    time_intervals : list
        List of time intervals from the first image
    max_shift : int
        Maximum shift in pixels
    step : int
        Step size between shifts
    method : str
        Stacking method
    
    Returns:
    --------
    dict : Dictionary with results keyed by (dx, dy) tuples
    """
    # Generate grid of shifts
    shifts_x = range(-max_shift, max_shift + 1, step)
    shifts_y = range(-max_shift, max_shift + 1, step)
    
    # Generate all motion vectors (combinations of x and y shifts)
    motion_vectors = [(dx, dy) for dx in shifts_x for dy in shifts_y]
    
    # Apply shift-and-stack for each motion vector
    results = {}
    
    for motion_vector in motion_vectors:
        stacked, shifts = shift_and_stack(images, motion_vector, time_intervals, method)
        results[motion_vector] = {
            'stacked': stacked,
            'shifts': shifts
        }
    
    return results

def apply_filter_to_stack(stacked_image, filter_type='matched', kernel_size=3):
    """
    Apply a filter to enhance faint point sources in stacked image
    
    Parameters:
    -----------
    stacked_image : numpy.ndarray
        Stacked image
    filter_type : str
        Filter type: 'matched', 'mexican_hat', 'gaussian', or 'median'
    kernel_size : int
        Size of the filter kernel
    
    Returns:
    --------
    numpy.ndarray : Filtered image
    """
    # Make a copy of the input to avoid modification
    filtered = np.copy(stacked_image)
    
    # Replace NaNs with zeros for filtering
    mask = np.isnan(filtered)
    filtered[mask] = 0
    
    if filter_type == 'matched':
        # Simple matched filter for point sources (Gaussian PSF)
        sigma = kernel_size / 6.0  # Approximate conversion
        filtered = ndimage.gaussian_filter(filtered, sigma)
    
    elif filter_type == 'mexican_hat':
        # Mexican hat filter (good for point sources)
        sigma = kernel_size / 6.0
        
        # Create Mexican hat kernel
        y, x = np.indices((kernel_size, kernel_size)) - (kernel_size - 1) / 2
        r = np.sqrt(x**2 + y**2)
        kernel = (1 - (r/sigma)**2) * np.exp(-(r**2)/(2*sigma**2))
        kernel = kernel - np.mean(kernel)  # Make sure kernel sums to zero
        
        # Apply filter via convolution
        filtered = ndimage.convolve(filtered, kernel)
    
    elif filter_type == 'gaussian':
        # Gaussian smoothing
        sigma = kernel_size / 6.0
        filtered = ndimage.gaussian_filter(filtered, sigma)
    
    elif filter_type == 'median':
        # Median filter
        filtered = ndimage.median_filter(filtered, size=kernel_size)
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Restore NaN mask
    filtered[mask] = np.nan
    
    return filtered

def combine_stacks(stacks, weights=None, method='mean'):
    """
    Combine multiple stacked images
    
    Parameters:
    -----------
    stacks : list
        List of stacked images
    weights : list or None
        List of weights for each stack
    method : str
        Combination method: 'mean', 'median', 'sum', or 'weighted_mean'
    
    Returns:
    --------
    numpy.ndarray : Combined stack
    """
    if weights is not None and method == 'weighted_mean':
        if len(weights) != len(stacks):
            raise ValueError("Number of weights must match number of stacks")
        
        # Calculate weighted mean
        combined = np.zeros_like(stacks[0])
        weight_sum = np.zeros_like(stacks[0])
        
        for i, stack in enumerate(stacks):
            mask = ~np.isnan(stack)
            combined[mask] += stack[mask] * weights[i]
            weight_sum[mask] += weights[i]
        
        # Normalize by sum of weights, avoiding division by zero
        mask = weight_sum > 0
        combined[mask] /= weight_sum[mask]
        combined[~mask] = np.nan
    
    elif method == 'mean':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            combined = np.nanmean(stacks, axis=0)
    
    elif method == 'median':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            combined = np.nanmedian(stacks, axis=0)
    
    elif method == 'sum':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            combined = np.nansum(stacks, axis=0)
    
    else:
        raise ValueError(f"Unknown combination method: {method}")
    
    return combined
def filter_sources(sources, min_snr=3.0, min_flux=None, max_sources=None, exclude_border=None):
    """
    Filter detected sources based on quality criteria
    
    Parameters:
    -----------
    sources : pandas.DataFrame
        Table of detected sources from detect_sources()
    min_snr : float
        Minimum signal-to-noise ratio for sources
    min_flux : float or None
        Minimum flux value for sources
    max_sources : int or None
        Maximum number of sources to return (sorted by flux)
    exclude_border : int or None
        Exclude sources this many pixels from the border
    
    Returns:
    --------
    pandas.DataFrame or None
        Filtered sources table, or None if no sources pass the filter
    """
    if sources is None or len(sources) == 0:
        return None
    
    # Make a copy of the sources table
    filtered = sources.copy()
    
    # Initial length
    initial_count = len(filtered)
    
    # Apply SNR filter if SNR column exists
    if 'snr' in filtered.colnames and min_snr is not None:
        filtered = filtered[filtered['snr'] >= min_snr]
    elif 'peak' in filtered.colnames and 'sky' in filtered.colnames and min_snr is not None:
        # Calculate SNR from peak and sky if possible
        filtered = filtered[filtered['peak'] / filtered['sky'] >= min_snr]
    
    # Apply flux filter
    if min_flux is not None and 'flux' in filtered.colnames:
        filtered = filtered[filtered['flux'] >= min_flux]
    
    # Apply border filter
    if exclude_border is not None:
        if 'xcentroid' in filtered.colnames and 'ycentroid' in filtered.colnames:
            # We need image dimensions for this filter, assume image is large enough
            # This is not ideal but will work in most cases
            margin = exclude_border
            filtered = filtered[(filtered['xcentroid'] >= margin) & 
                                (filtered['xcentroid'] <= 2048 - margin) &  # Assume standard image size
                                (filtered['ycentroid'] >= margin) & 
                                (filtered['ycentroid'] <= 2048 - margin)]
    
    # Sort by flux descending if flux column exists
    if 'flux' in filtered.colnames:
        filtered.sort('flux', reverse=True)
    elif 'peak' in filtered.colnames:
        filtered.sort('peak', reverse=True)
    
    # Limit number of sources
    if max_sources is not None and len(filtered) > max_sources:
        filtered = filtered[:max_sources]
    
    # If no sources pass the filter, return None
    if len(filtered) == 0:
        return None
    
    return filtered

def process_motion_vector(images, base_shifts, motion_vector, method='mean', detection_threshold=5.0):
    """
    Process a single motion vector on a set of images
    
    Parameters:
    -----------
    images : list
        List of 2D image arrays
    base_shifts : list
        List of (dx, dy) base shifts for each image (e.g. from initial alignment)
    motion_vector : tuple
        (dx, dy) motion vector to test, in pixels per image
    method : str
        Stacking method: 'mean', 'median', 'sum', or 'weighted_mean'
    detection_threshold : float
        Threshold for source detection in sigma
    
    Returns:
    --------
    tuple
        (stacked_image, detected_sources, shifts_applied)
    """
    # Calculate shifts for this motion vector
    # First shift is always (0,0) for the reference image
    shifts = [base_shifts[0]]
    
    # For the remaining images, calculate shift based on motion vector and index
    for i in range(1, len(images)):
        dx = base_shifts[i][0] - (i * motion_vector[0])
        dy = base_shifts[i][1] - (i * motion_vector[1])
        shifts.append((dx, dy))
    
    # Stack the images with these shifts
    stacked = stack_images(images, shifts, method=method)
    
    # Import here to avoid circular imports
    try:
        from detection.source_finder import detect_sources
        sources = detect_sources(stacked, threshold=detection_threshold)
    except ImportError:
        # If we can't import source_finder, continue without source detection
        sources = None
    
    return stacked, sources, shifts

