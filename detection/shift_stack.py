"""
shift_stack.py - Core implementation of shift-and-stack algorithm for KBO detection

This module provides functions to shift and stack images according to hypothetical
motion vectors, which allows detection of moving objects that would be too faint
to detect in single frames.
"""

import numpy as np
from scipy import ndimage
import warnings
import traceback
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

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
    
    try:
        # Find minimum dimensions to ensure all shifts result in same-sized arrays
        common_shape = []
        for img, shift in zip(images, shifts):
            # Predict shape after shifting
            shifted_shape = [img.shape[0], img.shape[1]]
            common_shape.append(shifted_shape)
        
        # Convert to numpy array and get minimum dimensions
        common_shape = np.array(common_shape)
        min_height, min_width = np.min(common_shape, axis=0)
        
        # Apply shifts and crop to common dimensions
        for i, (image, shift) in enumerate(zip(images, shifts)):
            try:
                dx, dy = shift
                shifted = apply_shift(image, dx, dy)
                
                # Crop to common dimensions
                shifted = shifted[:min_height, :min_width]
                
                shifted_images.append(shifted)
            except Exception as e:
                print(f"Error shifting image {i}: {e}")
                # Create placeholder of correct shape
                placeholder = np.full((min_height, min_width), np.nan)
                shifted_images.append(placeholder)
        
        # Convert to numpy array for efficient stacking
        # Note: This will raise an error if dimensions are inconsistent
        shifted_images = np.array(shifted_images)
    
    except Exception as e:
        print(f"Error preparing images for stacking: {e}")
        # Fallback approach: apply shifts but don't try to stack as 3D array
        shifted_images = []
        
        for i, (image, shift) in enumerate(zip(images, shifts)):
            try:
                dx, dy = shift
                shifted = apply_shift(image, dx, dy)
                shifted_images.append(shifted)
            except Exception as e:
                print(f"Error shifting image {i}: {e}")
                shifted_images.append(np.full_like(image, np.nan))
    
    # Now stack the images according to method
    try:
        # Stack according to method, handling NaNs properly
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
            # Handle weighted mean with explicit loop to avoid array shape issues
            weights_array = np.array(weights)
            stacked = np.zeros_like(shifted_images[0])
            weight_sum = np.zeros_like(shifted_images[0])
            
            for i, img in enumerate(shifted_images):
                mask = ~np.isnan(img)
                stacked[mask] += img[mask] * weights_array[i]
                weight_sum[mask] += weights_array[i]
            
            # Normalize by sum of weights, avoiding division by zero
            mask = weight_sum > 0
            stacked[mask] /= weight_sum[mask]
            stacked[~mask] = np.nan
        else:
            raise ValueError(f"Unknown stacking method: {method}")
            
    except Exception as e:
        print(f"Error stacking images: {e}")
        traceback.print_exc()
        
        # Create an emergency fallback
        print("Using emergency fallback for stacking")
        
        # Use a more direct loop approach
        # Start with the first image as base
        if shifted_images:
            stacked = np.copy(shifted_images[0])
            valid_count = (~np.isnan(stacked)).astype(float)
            
            # Add other images
            for img in shifted_images[1:]:
                mask = ~np.isnan(img)
                stacked[mask] += img[mask]
                valid_count[mask] += 1
            
            # Normalize by count of valid pixels (mean)
            mask = valid_count > 0
            stacked[mask] /= valid_count[mask]
        else:
            # If no images at all, return array of NaNs
            stacked = np.full((10, 10), np.nan)  # Arbitrary small size
    
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

def process_motion_vector(images, base_shifts, motion_vector, detection_threshold=5.0, fwhm=3.0):
    """
    Process a single motion vector and detect sources in the stacked image
    
    Parameters:
    -----------
    images : list
        List of 2D image arrays
    base_shifts : list
        List of base shifts for each image
    motion_vector : tuple
        (dx, dy) motion vector to test
    detection_threshold : float
        Detection threshold in sigma for source finding
    fwhm : float
        Full width at half maximum for source finding
    
    Returns:
    --------
    tuple
        (stacked_image, sources, shifts)
    """
    try:
        # Calculate shifts for this motion vector
        dx, dy = motion_vector
        shifts = []
        
        for i, base_shift in enumerate(base_shifts):
            # Add motion vector to base shift
            shift_x = base_shift[0] + i * dx
            shift_y = base_shift[1] + i * dy
            shifts.append((shift_x, shift_y))
        
        # Stack the images
        stacked = stack_images(images, shifts)
        
        # Detect sources in stacked image
        sources = detect_sources(stacked, threshold=detection_threshold, fwhm=fwhm)
        
        return stacked, sources, shifts
    
    except Exception as e:
        print(f"Error processing motion vector: {e}")
        print(f"Motion vector: {motion_vector}")
        traceback.print_exc()
        
        # Return empty results
        return np.full_like(images[0], np.nan), None, [(0, 0) for _ in images]

def detect_sources(image, threshold=5.0, fwhm=3.0):
    """
    Detect point sources in an image
    
    Parameters:
    -----------
    image : numpy.ndarray
        2D image data
    threshold : float
        Detection threshold in sigma
    fwhm : float
        Full width at half maximum for source finding
    
    Returns:
    --------
    Table or None
        Table of detected sources
    """
    try:
        # Replace NaNs with median for detection
        masked_image = np.copy(image)
        nan_mask = np.isnan(masked_image)
        if np.all(nan_mask):
            return None  # All NaNs, can't detect anything
        
        # Get median of non-NaN values
        median_value = np.nanmedian(masked_image)
        masked_image[nan_mask] = median_value
        
        # Calculate statistics with sigma clipping
        mean, median, std = sigma_clipped_stats(masked_image, sigma=3.0)
        
        # Create a source finder
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
        
        # Find sources
        sources = daofind(masked_image - median)
        
        return sources
    
    except Exception as e:
        print(f"Error detecting sources: {e}")
        return None

def filter_sources(sources, min_snr=3.0, min_sharpness=0.2, max_sharpness=0.8):
    """
    Filter detected sources based on quality criteria
    
    Parameters:
    -----------
    sources : Table
        Table of sources from detect_sources
    min_snr : float
        Minimum signal-to-noise ratio
    min_sharpness : float
        Minimum sharpness
    max_sharpness : float
        Maximum sharpness
    
    Returns:
    --------
    Table
        Filtered table of sources
    """
    if sources is None or len(sources) == 0:
        return None
    
    # Calculate SNR if not present
    if 'snr' not in sources.colnames:
        if 'peak' in sources.colnames and 'sky' in sources.colnames:
            sources['snr'] = sources['peak'] / sources['sky']
    
    # Apply filters
    mask = np.ones(len(sources), dtype=bool)
    
    # SNR filter
    if 'snr' in sources.colnames:
        mask &= sources['snr'] >= min_snr
    
    # Sharpness filter
    if 'sharpness' in sources.colnames:
        mask &= (sources['sharpness'] >= min_sharpness) & (sources['sharpness'] <= max_sharpness)
    
    # Apply the mask
    filtered_sources = sources[mask]
    
    return filtered_sources

def score_candidate(candidate, original_images, original_shifts, search_radius=3):
    """
    Score a KBO candidate based on multiple criteria
    
    Parameters:
    -----------
    candidate : dict
        Candidate object information
    original_images : list
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
    
    # Check if candidate is near the edge of any image
    for image in original_images:
        if (x < 10 or x > image.shape[1] - 10 or
            y < 10 or y > image.shape[0] - 10):
            # Near edge, reduce score
            return 0.3
    
    # Check consistency of candidate across images
    detections = 0
    for i, image in enumerate(original_images):
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
    
    # Calculate score based on detection rate
    detection_rate = detections / len(original_images)
    score = 0.5 + 0.5 * detection_rate
    
    # Adjust score based on motion vector
    dx, dy = motion_vector
    motion_speed = np.sqrt(dx*dx + dy*dy)
    
    # Penalize no motion (static sources)
    if motion_speed < 0.05:
        score *= 0.5
    
    # Penalize extremely high motion (likely noise)
    if motion_speed > 2.0:
        score *= 0.8
    
    return score