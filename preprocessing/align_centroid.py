"""
Centroid-based image alignment for astronomical images.

This module provides functions to align images using the centroid positions
of bright sources (stars), which is particularly effective when there are
distinct point sources in the images.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
import time
import warnings

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
    start_time = time.time()
    
    if verbose:
        print("    Using centroid-based alignment")
    
    # Try different threshold levels if needed
    thresholds = [threshold, 0.8, 0.7, 0.6, 0.5]
    max_sources = [5, 10, 20, 30, 50]  # Try more sources with lower thresholds
    
    for i, (thresh, max_src) in enumerate(zip(thresholds, max_sources)):
        # Find centroids in reference image
        ref_sources = find_sources_by_threshold(ref_image, thresh, max_n=max_src)
        
        # Find centroids in target image
        target_sources = find_sources_by_threshold(target_image, thresh, max_n=max_src)
        
        if len(ref_sources) > 2 and len(target_sources) > 2:
            if verbose and i > 0:
                print(f"    Used fallback threshold {thresh} to find {len(ref_sources)} reference sources "
                      f"and {len(target_sources)} target sources")
            break
    
    if len(ref_sources) < 3 or len(target_sources) < 3:
        if verbose:
            print("    Warning: Not enough sources for reliable centroid alignment")
            print(f"    Found {len(ref_sources)} reference sources and {len(target_sources)} target sources")
            print("    Using the brightest sources only")
        
        # If we have at least one source in each, use those
        if len(ref_sources) > 0 and len(target_sources) > 0:
            ref_y, ref_x = ref_sources[0]
            target_y, target_x = target_sources[0]
            
            # Calculate shift
            x_shift = ref_x - target_x
            y_shift = ref_y - target_y
        else:
            # No sources found, can't align
            if verbose:
                print("    No sources found. Cannot align using centroids.")
            return target_image.copy(), (0, 0)
    else:
        # We have multiple sources - try to match patterns
        x_shift, y_shift = match_source_patterns(ref_sources, target_sources, verbose)
    
    # Apply shift
    aligned_data = ndimage.shift(target_image, (y_shift, x_shift), 
                              order=1, mode='constant', cval=np.nan)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"    Centroid alignment completed in {elapsed:.1f} seconds")
        print(f"    Alignment shift: dx={x_shift:.2f}, dy={y_shift:.2f} pixels")
    
    return aligned_data, (x_shift, y_shift)

def find_sources_by_threshold(image, threshold_fraction=0.9, min_size=3, max_n=10):
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
    if np.all(mask):
        return []  # All NaN image
    
    data[mask] = np.nanmin(data) if not np.all(np.isnan(data)) else 0
    
    # Calculate threshold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        max_val = np.nanmax(data)
    
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

def match_source_patterns(ref_sources, target_sources, verbose=True):
    """
    Match patterns of sources between two images to determine shift
    
    Parameters:
    -----------
    ref_sources : list
        List of (y, x) centroids from reference image
    target_sources : list
        List of (y, x) centroids from target image
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    tuple
        (x_shift, y_shift)
    """
    # Convert to numpy arrays
    ref_array = np.array(ref_sources)
    target_array = np.array(target_sources)
    
    # If we have more than 3 sources in each image, we can try pattern matching
    if len(ref_sources) >= 3 and len(target_sources) >= 3:
        # Method 1: Try matching distances between sources
        try:
            # Calculate all pairwise distances within each image
            ref_dist = cdist(ref_array, ref_array)
            target_dist = cdist(target_array, target_array)
            
            # Find the best matching pattern
            best_score = float('inf')
            best_ref_idx = 0
            best_target_idx = 0
            
            # Try each reference source as the anchor
            for i in range(len(ref_sources)):
                # Get distances from this source to all others
                ref_distances = ref_dist[i]
                
                # Try each target source as a potential match
                for j in range(len(target_sources)):
                    target_distances = target_dist[j]
                    
                    # Calculate a score (lower is better)
                    # We use the sum of squared differences between sorted distances
                    ref_sorted = np.sort(ref_distances)[:min(len(ref_distances), len(target_distances))]
                    target_sorted = np.sort(target_distances)[:min(len(ref_distances), len(target_distances))]
                    
                    score = np.sum((ref_sorted - target_sorted) ** 2)
                    
                    if score < best_score:
                        best_score = score
                        best_ref_idx = i
                        best_target_idx = j
            
            # Now we have the best matching pair of sources
            # Use them to calculate the shift
            best_ref = ref_array[best_ref_idx]
            best_target = target_array[best_target_idx]
            
            y_shift = best_ref[0] - best_target[0]
            x_shift = best_ref[1] - best_target[1]
            
            if verbose:
                print(f"    Pattern matching succeeded. Match score: {best_score:.2f}")
            
            return x_shift, y_shift
            
        except Exception as e:
            if verbose:
                print(f"    Pattern matching failed: {e}")
    
    # Fallback: Just use the brightest source in each
    if verbose:
        print("    Using brightest source for alignment")
    
    ref_y, ref_x = ref_sources[0]
    target_y, target_x = target_sources[0]
    
    # Calculate shift
    x_shift = ref_x - target_x
    y_shift = ref_y - target_y
    
    return x_shift, y_shift