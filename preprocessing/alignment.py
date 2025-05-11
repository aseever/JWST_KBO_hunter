"""
Image alignment utilities for JWST data preprocessing.

This module serves as a dispatcher for different alignment methods,
selecting and applying the appropriate technique based on the data
and specified alignment method.
"""

import numpy as np
import time
import warnings
import traceback

# Import alignment methods from separate modules
try:
    from preprocessing.align_correlation import align_by_correlation
except ImportError:
    align_by_correlation = None

try:
    from preprocessing.align_centroid import align_by_centroid
except ImportError:
    align_by_centroid = None

try:
    from preprocessing.align_wcs import align_by_wcs
except ImportError:
    align_by_wcs = None


def align_images(images, reference_idx=0, method='centroid', verbose=True):
    """
    Align images to a common reference frame
    
    Parameters:
    -----------
    images : list
        List of image dictionaries from fits_loader
    reference_idx : int
        Index of the reference image
    method : str
        Alignment method: 'wcs', 'centroid', 'correlation', or 'none'
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
        print(f"  Requested alignment method: {method}")
    
    # Skip alignment if method is 'none'
    if method.lower() == 'none':
        if verbose:
            print("  Skipping alignment as requested")
        
        # Just copy cleaned data to aligned data
        for image in images:
            image['aligned_data'] = image['cleaned_data'].copy() if 'cleaned_data' in image else image['data'].copy()
            image['alignment_shift'] = (0, 0)  # No shift
        
        return images
    
    # Check if the specified method is available
    method_available = {
        'correlation': align_by_correlation is not None,
        'centroid': align_by_centroid is not None,
        'wcs': align_by_wcs is not None
    }
    
    # Prioritize methods in this order: WCS, centroid, correlation
    preferred_order = ['wcs', 'centroid', 'correlation']
    
    # If user explicitly requested a method, try that first
    if method in method_available and method_available[method]:
        preferred_order.remove(method)
        preferred_order.insert(0, method)
    
    # Use the specified image as reference
    reference = images[reference_idx]
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
        
        # Create a safe copy of the data in case alignment fails
        image['aligned_data'] = image['cleaned_data'].copy() if 'cleaned_data' in image else image['data'].copy()
        image['alignment_shift'] = (0, 0)  # Default to no shift
        
        # Try each method in order until one succeeds
        success = False
        
        for try_method in preferred_order:
            if not method_available[try_method]:
                continue
                
            try:
                start_time = time.time()
                
                if verbose and try_method != method:
                    print(f"    Trying {try_method} alignment method...")
                
                if try_method == 'correlation':
                    # Correlation-based alignment
                    aligned_data, shift = align_by_correlation(
                        reference['cleaned_data'] if 'cleaned_data' in reference else reference['data'],
                        image['cleaned_data'] if 'cleaned_data' in image else image['data'],
                        verbose=verbose
                    )
                    
                elif try_method == 'centroid':
                    # Centroid-based alignment
                    aligned_data, shift = align_by_centroid(
                        reference['cleaned_data'] if 'cleaned_data' in reference else reference['data'],
                        image['cleaned_data'] if 'cleaned_data' in image else image['data'],
                        verbose=verbose
                    )
                    
                elif try_method == 'wcs':
                    # WCS-based alignment
                    aligned_data, shift = align_by_wcs(
                        reference,
                        image,
                        verbose=verbose
                    )
                
                # Update the image with aligned data
                image['aligned_data'] = aligned_data
                image['alignment_shift'] = shift
                
                elapsed = time.time() - start_time
                if verbose:
                    print(f"    {try_method.capitalize()} alignment succeeded in {elapsed:.1f} seconds")
                
                success = True
                break  # Stop trying other methods once one succeeds
                
            except Exception as e:
                if verbose:
                    print(f"    {try_method.capitalize()} alignment failed: {e}")
        
        if not success and verbose:
            print(f"    All alignment methods failed. Using unaligned image.")
        
        aligned_images.append(image)
    
    return aligned_images

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
    data_list = [img['aligned_data'] for img in aligned_images if 'aligned_data' in img]
    
    if not data_list:
        return None
    
    # Find common dimensions (for consistent stacking)
    shapes = np.array([d.shape for d in data_list])
    min_shape = np.min(shapes, axis=0)
    
    # Crop all images to common dimensions
    cropped_data = []
    for data in data_list:
        cropped = data[:min_shape[0], :min_shape[1]]
        cropped_data.append(cropped)
    
    # Stack according to method
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        if method == 'mean':
            stacked = np.nanmean(cropped_data, axis=0)
        elif method == 'median':
            stacked = np.nanmedian(cropped_data, axis=0)
        elif method == 'sum':
            stacked = np.nansum(cropped_data, axis=0)
        else:
            raise ValueError(f"Unknown stacking method: {method}")
    
    return stacked