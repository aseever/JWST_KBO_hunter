"""
Calibration and cleaning utilities for JWST data preprocessing.

This module handles background subtraction, bad pixel removal,
and other calibration steps for JWST image data.
"""

import numpy as np
from astropy.stats import sigma_clip
from scipy import ndimage

def subtract_background(image_data, verbose=True):
    """
    Subtract background from image using sigma-clipping statistics
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        2D image data
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    tuple
        (background-subtracted data, background level, background noise)
    """
    if verbose:
        print("Subtracting background...")
        print(f"  Initial data range: {np.nanmin(image_data):.3e} to {np.nanmax(image_data):.3e}")
    
    # Mask NaN values
    mask = np.isnan(image_data)
    
    # Use sigma-clipping to estimate background
    clipped_data = sigma_clip(image_data[~mask], sigma=3, maxiters=5)
    bg_mean = np.nanmean(clipped_data)
    bg_median = np.nanmedian(clipped_data)
    bg_std = np.nanstd(clipped_data)
    
    if verbose:
        print(f"  Background statistics:")
        print(f"    Mean: {bg_mean:.3e}")
        print(f"    Median: {bg_median:.3e}")
        print(f"    Std Dev: {bg_std:.3e}")
    
    # Subtract background (use median as it's more robust)
    bg_subtracted = image_data - bg_median
    
    if verbose:
        print(f"  After background subtraction: {np.nanmin(bg_subtracted):.3e} to {np.nanmax(bg_subtracted):.3e}")
    
    return bg_subtracted, bg_median, bg_std

def clean_image(image_data, sigma=5, max_iters=3, filter_size=5, verbose=True):
    """
    Clean image by removing bad pixels, cosmic rays, etc.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        2D image data
    sigma : float
        Sigma threshold for outlier detection
    max_iters : int
        Maximum number of iterations for sigma clipping
    filter_size : int
        Size of median filter for bad pixel replacement
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    tuple
        (cleaned data, bad pixel mask)
    """
    if verbose:
        print("Cleaning image...")
    
    # Identify bad pixels using sigma clipping
    # We use a high sigma to avoid removing real sources
    masked_data = sigma_clip(image_data, sigma=sigma, maxiters=max_iters)
    
    # Count bad pixels
    bad_pixel_mask = masked_data.mask
    num_bad_pixels = np.sum(bad_pixel_mask)
    
    if verbose:
        print(f"  Identified {num_bad_pixels} bad pixels "
              f"({num_bad_pixels/image_data.size*100:.2f}% of image)")
    
    # Replace bad pixels with local median
    cleaned_data = np.copy(image_data)
    if num_bad_pixels > 0:
        # Use a median filter only on bad pixels
        median_filtered = ndimage.median_filter(image_data, size=filter_size)
        cleaned_data[bad_pixel_mask] = median_filtered[bad_pixel_mask]
    
    return cleaned_data, bad_pixel_mask

def apply_flat_field(image_data, flat_field, verbose=True):
    """
    Apply flat field correction to an image
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        2D image data to correct
    flat_field : numpy.ndarray
        Flat field image
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    numpy.ndarray
        Flat-field corrected image
    """
    if image_data.shape != flat_field.shape:
        raise ValueError("Image and flat field must have the same shape")
    
    # Normalize flat field
    norm_flat = flat_field / np.nanmedian(flat_field)
    
    # Apply flat field correction
    corrected = image_data / norm_flat
    
    if verbose:
        print("Applied flat field correction")
        print(f"  Original range: {np.nanmin(image_data):.3e} to {np.nanmax(image_data):.3e}")
        print(f"  Corrected range: {np.nanmin(corrected):.3e} to {np.nanmax(corrected):.3e}")
    
    return corrected

def normalize_image(image_data, method='minmax', verbose=True):
    """
    Normalize image data
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        2D image data to normalize
    method : str
        Normalization method: 'minmax', 'zscore', or 'robust'
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    numpy.ndarray
        Normalized image
    """
    if verbose:
        print(f"Normalizing image using {method} method")
    
    # Make a copy to avoid modifying the original
    normalized = np.copy(image_data)
    
    # Handle NaN values
    mask = np.isnan(normalized)
    
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.nanmin(normalized)
        max_val = np.nanmax(normalized)
        normalized = (normalized - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        # Z-score normalization
        mean = np.nanmean(normalized)
        std = np.nanstd(normalized)
        normalized = (normalized - mean) / std
    
    elif method == 'robust':
        # Robust normalization using percentiles
        p1, p99 = np.nanpercentile(normalized, [1, 99])
        normalized = (normalized - p1) / (p99 - p1)
        # Clip to [0, 1]
        normalized = np.clip(normalized, 0, 1)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Restore NaN values
    normalized[mask] = np.nan
    
    if verbose:
        print(f"  Normalized range: {np.nanmin(normalized):.3e} to {np.nanmax(normalized):.3e}")
    
    return normalized

def calculate_noise_properties(image_data, box_size=32, verbose=True):
    """
    Calculate noise properties of an image using a grid of boxes
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        2D image data
    box_size : int
        Size of boxes for local noise estimation
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    dict
        Dictionary with noise properties
    """
    ny, nx = image_data.shape
    
    # Create grid of boxes
    n_boxes_y = ny // box_size
    n_boxes_x = nx // box_size
    
    if verbose:
        print(f"Calculating noise properties using {n_boxes_y}×{n_boxes_x} grid of {box_size}×{box_size} boxes")
    
    # Collect statistics for each box
    box_stats = []
    
    for i in range(n_boxes_y):
        for j in range(n_boxes_x):
            y_start = i * box_size
            x_start = j * box_size
            
            # Extract box
            box = image_data[y_start:y_start+box_size, x_start:x_start+box_size]
            
            # Skip boxes with too many NaN values
            if np.sum(~np.isnan(box)) < box_size*box_size * 0.5:
                continue
            
            # Calculate statistics
            mean = np.nanmean(box)
            median = np.nanmedian(box)
            std = np.nanstd(box)
            
            box_stats.append({
                'y': y_start + box_size // 2,
                'x': x_start + box_size // 2,
                'mean': mean,
                'median': median,
                'std': std
            })
    
    # Calculate global statistics
    means = np.array([s['mean'] for s in box_stats])
    medians = np.array([s['median'] for s in box_stats])
    stds = np.array([s['std'] for s in box_stats])
    
    # Use sigma clipping to get robust estimates
    clipped_stds = sigma_clip(stds, sigma=3, maxiters=3)
    
    # Final noise estimate
    noise_estimate = np.nanmedian(clipped_stds)
    
    results = {
        'global_noise': noise_estimate,
        'min_noise': np.nanmin(stds),
        'max_noise': np.nanmax(stds),
        'mean_noise': np.nanmean(stds),
        'median_noise': np.nanmedian(stds),
        'noise_map': box_stats
    }
    
    if verbose:
        print(f"  Global noise estimate: {noise_estimate:.3e}")
        print(f"  Noise range: {results['min_noise']:.3e} to {results['max_noise']:.3e}")
    
    return results