"""
Image alignment utilities for JWST data preprocessing.

This module handles alignment of JWST images to a common reference frame,
which is essential for shift-and-stack KBO detection.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import correlate2d
import warnings

def align_images(images, reference_idx=0, method='correlation', verbose=True):
    """
    Align images to a common reference frame
    
    Parameters:
    -----------
    images : list
        List of image dictionaries from fits_loader
    reference_idx : int
        Index of the reference image
    method : str
        Alignment method: 'correlation', 'centroid', or 'wcs'
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    list
        List of aligned image dictionaries
    """
    if verbose:
        print("\nAligning images...")
        print(f"  Using image {reference_idx} as reference")
        print(f"  Alignment method: {method}")
    
    # Use the specified image as reference
    reference = images[reference_idx]
    ref_shape = reference['data'].shape
    
    aligned_images = []
    
    for i, image in enumerate(images):
        if i == reference_idx:
            # Reference image doesn't need alignment
            image['aligned_data'] = image['cleaned_data'] if 'cleaned_data' in image else image['data']
            image['alignment_shift'] = (0, 0)  # No shift
            aligned_images.append(image)
            continue
        
        if verbose:
            print(f"  Aligning image {i} to reference...")
        
        # Select alignment method
        if method == 'correlation':
            aligned_data, shift = align_by_correlation(
                reference['cleaned_data'] if 'cleaned_data' in reference else reference['data'],
                image['cleaned_data'] if 'cleaned_data' in image else image['data'],
                verbose=verbose
            )
        elif method == 'centroid':
            aligned_data, shift = align_by_centroid(
                reference['cleaned_data'] if 'cleaned_data' in reference else reference['data'],
                image['cleaned_data'] if 'cleaned_data' in image else image['data'],
                verbose=verbose
            )
        elif method == 'wcs':
            aligned_data, shift = align_by_wcs(
                reference,
                image,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown alignment method: {method}")
        
        image['aligned_data'] = aligned_data
        image['alignment_shift'] = shift
        aligned_images.append(image)
    
    return aligned_images

def align_by_correlation(ref_image, target_image, max_shift=50, verbose=True):
    """
    Align images using cross-correlation
    
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
    # Replace NaNs with zeros for correlation
    ref_filled = np.nan_to_num(ref_image, nan=0.0)
    target_filled = np.nan_to_num(target_image, nan=0.0)
    
    # Calculate cross-correlation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corr = correlate2d(ref_filled, target_filled, mode='same', boundary='symm')
    
    # Find peak correlation
    max_y, max_x = np.unravel_index(np.argmax(corr), corr.shape)
    
    # Calculate shift from center
    y_shift = max_y - (ref_image.shape[0] // 2)
    x_shift = max_x - (ref_image.shape[1] // 2)
    
    # Limit maximum shift
    if abs(x_shift) > max_shift or abs(y_shift) > max_shift:
        if verbose:
            print(f"    Warning: Calculated shift ({x_shift}, {y_shift}) exceeds maximum allowed shift {max_shift}")
            print(f"    Limiting shift to maximum allowed")
        
        x_shift = np.clip(x_shift, -max_shift, max_shift)
        y_shift = np.clip(y_shift, -max_shift, max_shift)
    
    # Apply shift
    aligned_data = ndimage.shift(target_image, (-y_shift, -x_shift), 
                              order=1, mode='constant', cval=np.nan)
    
    if verbose:
        print(f"    Alignment shift: dx={x_shift}, dy={y_shift} pixels")
    
    return aligned_data, (x_shift, y_shift)

def align_by_centroid(ref_image, target_image, threshold=0.9, verbose=True):
    """
    Align images by matching centroids of brightest sources
    
    Parameters:
    -----------
    ref_image : numpy.ndarray
        Reference image data
    target_image : numpy.ndarray
        Target image data to align
    threshold : float
        Threshold relative to maximum value for source detection
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    tuple
        (aligned image data, (x_shift, y_shift))
    """
    # Find centroids in reference image
    ref_sources = find_sources_by_threshold(ref_image, threshold)
    
    # Find centroids in target image
    target_sources = find_sources_by_threshold(target_image, threshold)
    
    if not ref_sources or not target_sources:
        if verbose:
            print("    Warning: Could not find sources for centroid alignment")
            print("    Falling back to correlation alignment")
        
        return align_by_correlation(ref_image, target_image, verbose=verbose)
    
    # Use brightest source in each image
    ref_y, ref_x = ref_sources[0]
    target_y, target_x = target_sources[0]
    
    # Calculate shift
    x_shift = ref_x - target_x
    y_shift = ref_y - target_y
    
    # Apply shift
    aligned_data = ndimage.shift(target_image, (y_shift, x_shift), 
                              order=1, mode='constant', cval=np.nan)
    
    if verbose:
        print(f"    Alignment shift: dx={x_shift}, dy={y_shift} pixels")
    
    return aligned_data, (x_shift, y_shift)

def align_by_wcs(ref_image_dict, target_image_dict, verbose=True):
    """
    Align images using WCS information
    
    Parameters:
    -----------
    ref_image_dict : dict
        Reference image dictionary from fits_loader
    target_image_dict : dict
        Target image dictionary to align
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    tuple
        (aligned image data, (x_shift, y_shift))
    """
    # Check if WCS is available
    if not ref_image_dict.get('wcs') or not target_image_dict.get('wcs'):
        if verbose:
            print("    Warning: WCS information not available for alignment")
            print("    Falling back to correlation alignment")
        
        ref_data = ref_image_dict['cleaned_data'] if 'cleaned_data' in ref_image_dict else ref_image_dict['data']
        target_data = target_image_dict['cleaned_data'] if 'cleaned_data' in target_image_dict else target_image_dict['data']
        
        return align_by_correlation(ref_data, target_data, verbose=verbose)
    
    try:
        # Get WCS objects
        ref_wcs = ref_image_dict['wcs']
        target_wcs = target_image_dict['wcs']
        
        # Get reference image data
        ref_data = ref_image_dict['cleaned_data'] if 'cleaned_data' in ref_image_dict else ref_image_dict['data']
        target_data = target_image_dict['cleaned_data'] if 'cleaned_data' in target_image_dict else target_image_dict['data']
        
        # Create a grid of pixel coordinates for the reference image
        ny, nx = ref_data.shape
        iy, ix = np.mgrid[:ny, :nx]
        
        # Check WCS dimension (MIRI sometimes has 3D WCS)
        if hasattr(ref_wcs, 'pixel_n_dim') and ref_wcs.pixel_n_dim == 3:
            # Create dummy wavelength index for 3D WCS
            if verbose:
                print("    Using 3D WCS with central wavelength index")
            
            # Use central wavelength index
            if hasattr(ref_image_dict, 'data') and len(ref_image_dict['data'].shape) == 3:
                wl_idx = ref_image_dict['data'].shape[0] // 2
            else:
                wl_idx = 0
            
            # Use 3D WCS
            try:
                # Convert reference pixel coordinates to sky coordinates
                ra, dec = ref_wcs.pixel_to_world(np.full_like(ix, wl_idx), iy, ix)
                
                # Convert sky coordinates to pixel coordinates in the target image
                if hasattr(target_wcs, 'pixel_n_dim') and target_wcs.pixel_n_dim == 3:
                    _, target_y, target_x = target_wcs.world_to_pixel(ra, dec)
                else:
                    target_y, target_x = target_wcs.world_to_pixel(ra, dec)
            except Exception as e:
                if verbose:
                    print(f"    WCS transformation error: {e}")
                    print("    Falling back to correlation alignment")
                
                return align_by_correlation(ref_data, target_data, verbose=verbose)
        else:
            try:
                # 2D WCS case
                # Convert reference pixel coordinates to sky coordinates
                ra, dec = ref_wcs.pixel_to_world(ix, iy)
                
                # Convert sky coordinates to pixel coordinates in the target image
                target_x, target_y = target_wcs.world_to_pixel(ra, dec)
            except Exception as e:
                if verbose:
                    print(f"    WCS transformation error: {e}")
                    print("    Falling back to correlation alignment")
                
                return align_by_correlation(ref_data, target_data, verbose=verbose)
        
        # Calculate median shift
        x_shifts = target_x - ix
        y_shifts = target_y - iy
        
        x_shift = np.nanmedian(x_shifts)
        y_shift = np.nanmedian(y_shifts)
        
        # Check for problematic shifts
        if not np.isfinite(x_shift) or not np.isfinite(y_shift):
            if verbose:
                print("    Warning: Non-finite shifts from WCS alignment")
                print("    Falling back to correlation alignment")
            
            return align_by_correlation(ref_data, target_data, verbose=verbose)
        
        # Apply shift
        aligned_data = ndimage.shift(target_data, (y_shift, x_shift), 
                                  order=1, mode='constant', cval=np.nan)
        
        if verbose:
            print(f"    WCS alignment shift: dx={x_shift:.2f}, dy={y_shift:.2f} pixels")
        
        return aligned_data, (x_shift, y_shift)
    
    except Exception as e:
        if verbose:
            print(f"    WCS alignment error: {e}")
            print("    Falling back to correlation alignment")
        
        ref_data = ref_image_dict['cleaned_data'] if 'cleaned_data' in ref_image_dict else ref_image_dict['data']
        target_data = target_image_dict['cleaned_data'] if 'cleaned_data' in target_image_dict else target_image_dict['data']
        
        return align_by_correlation(ref_data, target_data, verbose=verbose)

def find_sources_by_threshold(image, threshold_fraction=0.9, min_size=3, max_n=5):
    """
    Find sources in an image using simple thresholding
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image data
    threshold_fraction : float
        Fraction of maximum value to use as threshold
    min_size : int
        Minimum size of source in pixels
    max_n : int
        Maximum number of sources to return
    
    Returns:
    --------
    list
        List of (y, x) centroids sorted by brightness
    """
    # Copy and handle NaNs
    data = np.copy(image)
    mask = np.isnan(data)
    data[mask] = np.nanmin(data)
    
    # Calculate threshold
    max_val = np.max(data)
    threshold = threshold_fraction * max_val
    
    # Threshold the image
    binary = data > threshold
    
    # Label connected components
    labeled, num_features = ndimage.label(binary)
    
    if num_features == 0:
        return []
    
    # Measure properties of each component
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    mean_vals = ndimage.mean(data, labeled, range(1, num_features + 1))
    
    # Sort by brightness
    sorted_idx = np.argsort(-mean_vals)
    
    # Get centroids of brightest sources
    centroids = []
    for i in sorted_idx[:max_n]:
        if sizes[i] >= min_size:
            # Calculate centroid
            coords = np.where(labeled == i + 1)
            y_centroid = np.mean(coords[0])
            x_centroid = np.mean(coords[1])
            centroids.append((y_centroid, x_centroid))
    
    return centroids

def stack_aligned_images(aligned_images, method='mean'):
    """
    Stack aligned images
    
    Parameters:
    -----------
    aligned_images : list
        List of dictionaries with 'aligned_data' key
    method : str
        Stacking method: 'mean', 'median', or 'sum'
    
    Returns:
    --------
    numpy.ndarray
        Stacked image
    """
    # Extract aligned data
    data_list = [img['aligned_data'] for img in aligned_images]
    
    # Create a 3D array
    data_cube = np.stack(data_list, axis=0)
    
    # Stack according to method
    if method == 'mean':
        stacked = np.nanmean(data_cube, axis=0)
    elif method == 'median':
        stacked = np.nanmedian(data_cube, axis=0)
    elif method == 'sum':
        stacked = np.nansum(data_cube, axis=0)
    else:
        raise ValueError(f"Unknown stacking method: {method}")
    
    return stacked