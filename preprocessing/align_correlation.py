"""
Correlation-based image alignment for astronomical images.

This module provides functions to align images using cross-correlation,
which is effective for aligning images with extended features.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import time
import warnings

def align_by_correlation(ref_image, target_image, max_shift=50, verbose=True):
    """
    Align images using cross-correlation with ROI approach
    
    Parameters:
    -----------
    ref_image : numpy.ndarray
        Reference image data
    target_image : numpy.ndarray
        Target image data to align
    max_shift : int
        Maximum allowed shift in pixels
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    tuple
        (aligned image data, (x_shift, y_shift))
    """
    if verbose:
        print("    Using ROI-based correlation alignment")
        print("    TODO: Implement full multi-resolution correlation for better precision")
    
    # SIMPLE VERSION: Only use the central region for correlation
    # This greatly reduces computation while capturing the most important part
    
    # Extract central regions from both images
    h, w = ref_image.shape
    
    # Use a reasonably sized region (max 1000x1000 pixels)
    roi_size = min(1000, min(h, w) - 20)
    
    # Calculate center regions
    y_center, x_center = h // 2, w // 2
    half_roi = roi_size // 2
    
    y_min = max(0, y_center - half_roi)
    y_max = min(h, y_center + half_roi)
    x_min = max(0, x_center - half_roi)
    x_max = min(w, x_center + half_roi)
    
    ref_roi = ref_image[y_min:y_max, x_min:x_max]
    target_roi = target_image[y_min:y_max, x_min:x_max]
    
    if verbose:
        print(f"    Using central {roi_size}x{roi_size} region for correlation")
    
    # Perform direct correlation on the ROIs
    start_time = time.time()
    
    # Replace NaNs with zeros for correlation
    ref_filled = np.nan_to_num(ref_roi, nan=0.0)
    target_filled = np.nan_to_num(target_roi, nan=0.0)
    
    # Calculate cross-correlation using FFT (much faster for large images)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Use 'same' mode to get a correlation the same size as the inputs
        corr = fftconvolve(ref_filled, target_filled[::-1, ::-1], mode='same')
    
    # Find peak correlation
    max_y, max_x = np.unravel_index(np.argmax(corr), corr.shape)
    
    # Calculate shift from center
    y_shift = max_y - (ref_roi.shape[0] // 2)
    x_shift = max_x - (ref_roi.shape[1] // 2)
    
    # Apply shift to the full image
    aligned_data = ndimage.shift(target_image, (-y_shift, -x_shift), 
                              order=1, mode='constant', cval=np.nan)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"    Correlation completed in {elapsed:.1f} seconds")
        print(f"    Correlation shift: dx={x_shift}, dy={y_shift} pixels")
    
    return aligned_data, (x_shift, y_shift)

def align_by_correlation_roi(ref_image, target_image, roi_size=1000, verbose=True):
    """
    Perform correlation alignment using regions of interest
    
    Parameters:
    -----------
    ref_image : numpy.ndarray
        Reference image data
    target_image : numpy.ndarray
        Target image data to align
    roi_size : int
        Size of the region of interest to use
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    tuple
        (x_shift, y_shift)
    """
    # Extract regions of interest from the center of each image
    h, w = ref_image.shape
    
    # Make sure roi_size isn't larger than the image
    roi_size = min(roi_size, min(h, w) - 20)
    
    # Calculate center regions
    y_center, x_center = h // 2, w // 2
    half_roi = roi_size // 2
    
    y_min = max(0, y_center - half_roi)
    y_max = min(h, y_center + half_roi)
    x_min = max(0, x_center - half_roi)
    x_max = min(w, x_center + half_roi)
    
    ref_roi = ref_image[y_min:y_max, x_min:x_max]
    target_roi = target_image[y_min:y_max, x_min:x_max]
    
    if verbose:
        print(f"    Using ROI of size {ref_roi.shape} for initial alignment")
    
    # Perform correlation on the ROIs
    _, (x_shift, y_shift) = align_by_correlation_direct(ref_roi, target_roi, verbose=False)
    
    if verbose:
        print(f"    Initial ROI alignment shift: dx={x_shift:.2f}, dy={y_shift:.2f} pixels")
    
    return x_shift, y_shift

def refine_correlation_shift(ref_image, target_image, initial_shift, search_window=10, verbose=True):
    """
    Refine a correlation shift using a local search window
    
    Parameters:
    -----------
    ref_image : numpy.ndarray
        Reference image data
    target_image : numpy.ndarray
        Target image data to align
    initial_shift : tuple
        Initial (x_shift, y_shift) estimate
    search_window : int
        Size of the local search window in pixels
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    tuple
        (x_shift, y_shift)
    """
    # Apply the initial shift to get a rough alignment
    init_x, init_y = initial_shift
    rough_aligned = ndimage.shift(target_image, (-init_y, -init_x), 
                               order=1, mode='constant', cval=np.nan)
    
    # Extract regions for local refinement
    h, w = ref_image.shape
    
    # Calculate central region for correlation
    # Use a region that's 3 times the search window to ensure good correlation
    region_size = search_window * 3
    region_size = min(region_size, min(h, w) // 2)  # Make sure it's not too large
    
    y_center, x_center = h // 2, w // 2
    half_region = region_size // 2
    
    y_min = max(0, y_center - half_region)
    y_max = min(h, y_center + half_region)
    x_min = max(0, x_center - half_region)
    x_max = min(w, x_center + half_region)
    
    ref_region = ref_image[y_min:y_max, x_min:x_max]
    target_region = rough_aligned[y_min:y_max, x_min:x_max]
    
    if verbose:
        print(f"    Refining alignment using {region_size}x{region_size} center region")
    
    # Calculate local correlation to refine the shift
    _, (local_x, local_y) = align_by_correlation_direct(ref_region, target_region, 
                                                     max_shift=search_window,
                                                     verbose=False)
    
    # Combine the initial and refinement shifts
    final_x = init_x + local_x
    final_y = init_y + local_y
    
    if verbose:
        print(f"    Refinement shift: dx={local_x:.2f}, dy={local_y:.2f} pixels")
        print(f"    Combined shift: dx={final_x:.2f}, dy={final_y:.2f} pixels")
    
    return final_x, final_y

def align_by_correlation_direct(ref_image, target_image, max_shift=50, verbose=True):
    """
    Perform direct correlation alignment using FFT
    
    Parameters:
    -----------
    ref_image : numpy.ndarray
        Reference image data
    target_image : numpy.ndarray
        Target image data to align
    max_shift : int
        Maximum allowed shift in pixels
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    tuple
        (aligned image data, (x_shift, y_shift))
    """
    start_time = time.time()
    
    # Replace NaNs with zeros for correlation
    ref_filled = np.nan_to_num(ref_image, nan=0.0)
    target_filled = np.nan_to_num(target_image, nan=0.0)
    
    # Calculate cross-correlation using FFT (much faster for large images)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Use 'same' mode to get a correlation the same size as the inputs
        corr = fftconvolve(ref_filled, target_filled[::-1, ::-1], mode='same')
    
    # Find peak correlation
    max_y, max_x = np.unravel_index(np.argmax(corr), corr.shape)
    
    # Calculate shift from center
    y_shift = max_y - (ref_image.shape[0] // 2)
    x_shift = max_x - (ref_image.shape[1] // 2)
    
    # Apply shift
    aligned_data = ndimage.shift(target_image, (-y_shift, -x_shift), 
                              order=1, mode='constant', cval=np.nan)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"    Direct correlation completed in {elapsed:.1f} seconds")
        print(f"    Correlation shift: dx={x_shift}, dy={y_shift} pixels")
    
    return aligned_data, (x_shift, y_shift)

def visualize_correlation(ref_image, target_image, corr, shift):
    """
    Visualize the correlation result
    
    Parameters:
    -----------
    ref_image : numpy.ndarray
        Reference image data
    target_image : numpy.ndarray
        Target image data to align
    corr : numpy.ndarray
        Correlation matrix
    shift : tuple
        (x_shift, y_shift)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Reference image
    vmin, vmax = np.nanpercentile(ref_image, [1, 99])
    axes[0, 0].imshow(ref_image, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Reference Image')
    
    # Target image
    vmin, vmax = np.nanpercentile(target_image, [1, 99])
    axes[0, 1].imshow(target_image, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Target Image')
    
    # Correlation
    vmin, vmax = np.nanpercentile(corr, [1, 99])
    im = axes[1, 0].imshow(corr, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Correlation')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Mark peak correlation
    center_y, center_x = corr.shape[0] // 2, corr.shape[1] // 2
    peak_y, peak_x = center_y + shift[1], center_x + shift[0]
    axes[1, 0].plot(peak_x, peak_y, 'rx', markersize=10)
    
    # Aligned image
    aligned = ndimage.shift(target_image, (-shift[1], -shift[0]), 
                          order=1, mode='constant', cval=np.nan)
    vmin, vmax = np.nanpercentile(aligned, [1, 99])
    axes[1, 1].imshow(aligned, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'Aligned Image (dx={shift[0]:.1f}, dy={shift[1]:.1f})')
    
    plt.tight_layout()
    return fig