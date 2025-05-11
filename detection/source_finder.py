"""
source_finder.py - Detect sources in stacked images for KBO hunting

This module provides functions to detect point sources in stacked images,
measure their properties, and evaluate candidate KBOs based on signal strength,
position, and appearance in multiple stacked images.
"""

import numpy as np
from scipy import ndimage
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import warnings

def detect_sources(image, threshold=5.0, fwhm=3.0, exclude_border=10):
    """
    Detect point sources in an image using DAOStarFinder
    
    Parameters:
    -----------
    image : numpy.ndarray
        2D image data
    threshold : float
        Detection threshold in sigma above background
    fwhm : float
        FWHM of the PSF in pixels
    exclude_border : int
        Exclude sources this many pixels from the border
    
    Returns:
    --------
    pandas.DataFrame or None
        Table of detected sources, or None if no sources detected
    """
    # Copy the image to avoid modifying the original
    data = np.copy(image)
    
    # Replace NaNs with median for detection
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = np.isnan(data)
        if np.any(mask):
            # Get median of non-NaN values
            valid_data = data[~mask]
            if len(valid_data) > 0:
                data[mask] = np.median(valid_data)
            else:
                # All NaN image - return no sources
                return None
    
    # Calculate background statistics with sigma clipping
    try:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    except ValueError:
        # If statistical calculation fails, return no sources
        return None
    
    # Create a source finder
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
    
    # Find sources
    sources = daofind(data - median)
    
    # Handle case where no sources are found
    if sources is None or len(sources) == 0:
        return None
    
    # Filter by distance from border
    if exclude_border > 0:
        ny, nx = data.shape
        mask = ((sources['xcentroid'] > exclude_border) & 
                (sources['xcentroid'] < nx - exclude_border) &
                (sources['ycentroid'] > exclude_border) & 
                (sources['ycentroid'] < ny - exclude_border))
        
        sources = sources[mask]
        
        # Check if any sources remain after filtering
        if len(sources) == 0:
            return None
    
    return sources

def calculate_source_properties(image, sources, aperture_radius=5):
    """
    Calculate additional properties for detected sources
    
    Parameters:
    -----------
    image : numpy.ndarray
        2D image data
    sources : pandas.DataFrame
        Table of detected sources from detect_sources()
    aperture_radius : int
        Radius of aperture for flux measurement
    
    Returns:
    --------
    pandas.DataFrame
        Table of sources with additional properties
    """
    if sources is None or len(sources) == 0:
        return None
    
    # Make a copy of the sources table
    enhanced_sources = sources.copy()
    
    # Calculate additional properties for each source
    snr_values = []
    background_values = []
    aperture_flux_values = []
    
    for i, source in enumerate(sources):
        x, y = int(round(source['xcentroid'])), int(round(source['ycentroid']))
        
        # Extract region around source for local measurements
        x_min = max(0, x - aperture_radius)
        x_max = min(image.shape[1], x + aperture_radius + 1)
        y_min = max(0, y - aperture_radius)
        y_max = min(image.shape[0], y + aperture_radius + 1)
        
        region = image[y_min:y_max, x_min:x_max]
        
        # Skip if region is all NaN
        if np.all(np.isnan(region)):
            snr_values.append(np.nan)
            background_values.append(np.nan)
            aperture_flux_values.append(np.nan)
            continue
        
        # Calculate local background
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            background = np.nanmedian(region)
            background_std = np.nanstd(region)
        
        # Create aperture mask
        y_grid, x_grid = np.ogrid[:region.shape[0], :region.shape[1]]
        center_y = region.shape[0] // 2
        center_x = region.shape[1] // 2
        distance = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        aperture_mask = distance <= aperture_radius
        
        # Sum flux within aperture
        aperture_pixels = region[aperture_mask]
        valid_pixels = aperture_pixels[~np.isnan(aperture_pixels)]
        
        if len(valid_pixels) > 0:
            aperture_flux = np.sum(valid_pixels - background)
            
            # Calculate SNR
            pixel_noise = background_std
            aperture_noise = pixel_noise * np.sqrt(len(valid_pixels))
            snr = aperture_flux / aperture_noise if aperture_noise > 0 else 0
            
            snr_values.append(snr)
            background_values.append(background)
            aperture_flux_values.append(aperture_flux)
        else:
            snr_values.append(np.nan)
            background_values.append(np.nan)
            aperture_flux_values.append(np.nan)
    
    # Add new columns to the sources table
    enhanced_sources['snr'] = snr_values
    enhanced_sources['background'] = background_values
    enhanced_sources['aperture_flux'] = aperture_flux_values
    
    return enhanced_sources

def score_kbo_candidates(sources, motion_vector, image_shape, metrics=None):
    """
    Score KBO candidates based on various criteria
    
    Parameters:
    -----------
    sources : pandas.DataFrame
        Table of sources with properties
    motion_vector : tuple
        (dx, dy) motion vector that was used for the stack
    image_shape : tuple
        Shape of the stacked image (ny, nx)
    metrics : dict or None
        Custom scoring metrics to use
    
    Returns:
    --------
    pandas.DataFrame
        Sources with scores added
    """
    if sources is None or len(sources) == 0:
        return None
    
    # Make a copy of the sources table
    scored_sources = sources.copy()
    
    # Default metrics
    default_metrics = {
        'snr_weight': 1.0,           # Weight for SNR
        'sharpness_weight': 0.5,      # Weight for shape/sharpness
        'position_weight': 0.25,      # Weight for position (distance from edge)
        'flux_weight': 0.5            # Weight for flux
    }
    
    # Use provided metrics if available, otherwise use defaults
    if metrics is not None:
        for key, value in metrics.items():
            default_metrics[key] = value
    
    metrics = default_metrics
    
    # Calculate scores for each source
    snr_scores = []
    sharpness_scores = []
    position_scores = []
    flux_scores = []
    total_scores = []
    
    for i, source in enumerate(sources):
        # SNR score (primary factor)
        snr = source['snr'] if 'snr' in source.colnames else source['peak'] / source['sky']
        snr_score = min(1.0, snr / 20.0)  # Normalize to [0, 1] with cap at SNR=20
        
        # Sharpness score
        if 'sharpness' in source.colnames:
            sharpness = source['sharpness']
            # Ideal sharpness is close to 0.5 for point sources
            sharpness_score = 1.0 - abs(sharpness - 0.5) * 2.0
        else:
            sharpness_score = 0.5  # Default if not available
        
        # Position score (higher for sources away from edges)
        x, y = source['xcentroid'], source['ycentroid']
        nx, ny = image_shape[1], image_shape[0]
        
        # Distance from center as fraction of half-width
        dx = abs(x - nx/2) / (nx/2)
        dy = abs(y - ny/2) / (ny/2)
        
        # Penalize sources near the edge
        position_score = 1.0 - max(dx, dy)
        
        # Flux score (higher for brighter sources)
        if 'flux' in source.colnames:
            flux = source['flux']
            flux_max = np.max(sources['flux'])
            flux_score = flux / flux_max if flux_max > 0 else 0.5
        else:
            flux_score = 0.5  # Default if not available
        
        # Calculate total score
        total_score = (
            metrics['snr_weight'] * snr_score +
            metrics['sharpness_weight'] * sharpness_score +
            metrics['position_weight'] * position_score +
            metrics['flux_weight'] * flux_score
        ) / (metrics['snr_weight'] + metrics['sharpness_weight'] + 
             metrics['position_weight'] + metrics['flux_weight'])
        
        snr_scores.append(snr_score)
        sharpness_scores.append(sharpness_score)
        position_scores.append(position_score)
        flux_scores.append(flux_score)
        total_scores.append(total_score)
    
    # Add scores to sources table
    scored_sources['snr_score'] = snr_scores
    scored_sources['sharpness_score'] = sharpness_scores
    scored_sources['position_score'] = position_scores
    scored_sources['flux_score'] = flux_scores
    scored_sources['total_score'] = total_scores
    
    # Add motion vector for reference
    scored_sources['motion_dx'] = motion_vector[0]
    scored_sources['motion_dy'] = motion_vector[1]
    scored_sources['motion_speed'] = np.sqrt(motion_vector[0]**2 + motion_vector[1]**2)
    scored_sources['motion_angle'] = np.arctan2(motion_vector[1], motion_vector[0]) * 180 / np.pi
    
    # Sort by total score (descending)
    scored_sources.sort('total_score', reverse=True)
    
    return scored_sources

def find_source_in_original_images(source, original_images, shifts, search_radius=3):
    """
    Check if a source can be found in the original images
    
    Parameters:
    -----------
    source : dict-like
        Source properties including position
    original_images : list
        List of original image arrays
    shifts : list
        List of (dx, dy) shifts applied to each image
    search_radius : int
        Radius to search around expected position
    
    Returns:
    --------
    dict : Results of search
    """
    x_stack, y_stack = source['xcentroid'], source['ycentroid']
    detections = []
    
    for i, image in enumerate(original_images):
        # Calculate expected position in this image by reversing the shift
        dx, dy = shifts[i]
        x_img = x_stack - dx
        y_img = y_stack - dy
        
        # Skip if outside image bounds
        if (x_img < 0 or x_img >= image.shape[1] or 
            y_img < 0 or y_img >= image.shape[0]):
            detections.append({
                'detected': False,
                'out_of_bounds': True,
                'image_idx': i
            })
            continue
        
        # Extract region around expected position
        x_min = int(max(0, x_img - search_radius))
        x_max = int(min(image.shape[1], x_img + search_radius + 1))
        y_min = int(max(0, y_img - search_radius))
        y_max = int(min(image.shape[0], y_img + search_radius + 1))
        
        region = image[y_min:y_max, x_min:x_max]
        
        # Skip if region is all NaN
        if np.all(np.isnan(region)):
            detections.append({
                'detected': False,
                'all_nan': True,
                'image_idx': i
            })
            continue
        
        # Calculate local statistics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            region_median = np.nanmedian(region)
            region_std = np.nanstd(region)
        
        # Get value at expected position
        x_offset = int(x_img - x_min)
        y_offset = int(y_img - y_min)
        
        if (0 <= x_offset < region.shape[1] and 0 <= y_offset < region.shape[0]):
            pixel_value = region[y_offset, x_offset]
            
            if np.isnan(pixel_value):
                detections.append({
                    'detected': False,
                    'nan_pixel': True,
                    'image_idx': i
                })
                continue
            
            # Calculate signal-to-noise ratio
            snr = (pixel_value - region_median) / region_std if region_std > 0 else 0
            
            # Consider detected if SNR exceeds threshold
            detected = snr > 3.0
            
            detections.append({
                'detected': detected,
                'snr': snr,
                'value': pixel_value,
                'background': region_median,
                'image_idx': i,
                'x': x_img,
                'y': y_img
            })
        else:
            detections.append({
                'detected': False,
                'position_error': True,
                'image_idx': i
            })
    
    # Count detections
    num_detections = sum(1 for d in detections if d.get('detected', False))
    
    return {
        'source': source,
        'detections': detections,
        'num_detections': num_detections,
        'detection_fraction': num_detections / len(original_images) if original_images else 0
    }

def combine_detections_from_multiple_stacks(all_sources, tolerance=3.0):
    """
    Find duplicate detections across different stacks
    
    Parameters:
    -----------
    all_sources : list
        List of (sources, motion_vector, image_shape) tuples
    tolerance : float
        Maximum distance in pixels to consider sources the same
    
    Returns:
    --------
    dict : Groups of detections and their properties
    """
    if not all_sources:
        return {}
    
    # Extract all individual sources
    sources_list = []
    for sources, motion_vector, image_shape in all_sources:
        if sources is not None and len(sources) > 0:
            # Add motion vector to each source
            for i, source in enumerate(sources):
                sources_list.append({
                    'source': source,
                    'motion_vector': motion_vector,
                    'image_shape': image_shape,
                    'source_idx': i,
                    'stack_idx': len(sources_list)
                })
    
    if not sources_list:
        return {}
    
    # Group sources that are close to each other
    grouped_sources = []
    used_indices = set()
    
    for i, source_info in enumerate(sources_list):
        if i in used_indices:
            continue
        
        source = source_info['source']
        x1, y1 = source['xcentroid'], source['ycentroid']
        
        # Start a new group
        group = [source_info]
        used_indices.add(i)
        
        # Find all sources close to this one
        for j, other_info in enumerate(sources_list):
            if j == i or j in used_indices:
                continue
            
            other = other_info['source']
            x2, y2 = other['xcentroid'], other['ycentroid']
            
            # Calculate distance
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if distance <= tolerance:
                group.append(other_info)
                used_indices.add(j)
        
        grouped_sources.append(group)
    
    # Process groups and calculate properties
    results = {}
    
    for i, group in enumerate(grouped_sources):
        group_id = f"group_{i+1}"
        
        # Calculate average position
        x_sum = sum(info['source']['xcentroid'] for info in group)
        y_sum = sum(info['source']['ycentroid'] for info in group)
        x_avg = x_sum / len(group)
        y_avg = y_sum / len(group)
        
        # List all motion vectors
        motion_vectors = [info['motion_vector'] for info in group]
        
        # Calculate average scores
        if 'total_score' in group[0]['source'].colnames:
            avg_score = sum(info['source']['total_score'] for info in group) / len(group)
        else:
            avg_score = None
        
        results[group_id] = {
            'position': (x_avg, y_avg),
            'num_detections': len(group),
            'avg_score': avg_score,
            'detections': group,
            'motion_vectors': motion_vectors
        }
    
    return results
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

def score_candidate(candidate, original_images, shifts, search_radius=5):
    """
    Score a KBO candidate based on its appearance in the original images
    
    Parameters:
    -----------
    candidate : dict
        Candidate object with position and other properties
    original_images : list
        List of original unstacked images
    shifts : list
        List of (dx, dy) shifts for each image to follow the candidate's motion
    search_radius : int
        Radius around expected position to search in pixels
    
    Returns:
    --------
    float : Score between 0 and 1
    """
    # Extract candidate position
    x_center = candidate['xcentroid']
    y_center = candidate['ycentroid']
    
    # Initialize scoring metrics
    detection_count = 0
    snr_values = []
    position_errors = []
    
    # Check each original image
    for i, image in enumerate(original_images):
        # Get the expected position in this image by applying the inverse shift
        dx, dy = shifts[i]
        expected_x = x_center + dx
        expected_y = y_center + dy
        
        # Skip if the expected position is outside the image
        if (expected_x < 0 or expected_x >= image.shape[1] or
            expected_y < 0 or expected_y >= image.shape[0]):
            continue
        
        # Extract a small region around the expected position
        x_min = max(0, int(expected_x - search_radius))
        x_max = min(image.shape[1], int(expected_x + search_radius + 1))
        y_min = max(0, int(expected_y - search_radius))
        y_max = min(image.shape[0], int(expected_y + search_radius + 1))
        
        region = image[y_min:y_max, x_min:x_max]
        
        # Skip if region is all NaN
        if np.all(np.isnan(region)):
            continue
        
        # Calculate statistics for the region
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bg_median = np.nanmedian(region)
            bg_std = np.nanstd(region)
        
        # Get the value at the expected position
        expected_x_local = expected_x - x_min
        expected_y_local = expected_y - y_min
        
        # Find the maximum value near the expected position
        # Use a small window to find the brightest pixel
        window_radius = min(2, search_radius)
        window_x_min = max(0, int(expected_x_local - window_radius))
        window_x_max = min(region.shape[1], int(expected_x_local + window_radius + 1))
        window_y_min = max(0, int(expected_y_local - window_radius))
        window_y_max = min(region.shape[0], int(expected_y_local + window_radius + 1))
        
        window = region[window_y_min:window_y_max, window_x_min:window_x_max]
        
        if np.any(~np.isnan(window)):
            # Find the brightest non-NaN pixel
            max_value = np.nanmax(window)
            brightest_idx = np.nanargmax(window)
            brightest_y, brightest_x = np.unravel_index(brightest_idx, window.shape)
            
            # Calculate position relative to expected position
            pos_error = np.sqrt(
                (brightest_x + window_x_min - expected_x_local)**2 + 
                (brightest_y + window_y_min - expected_y_local)**2
            )
            
            # Calculate SNR
            snr = (max_value - bg_median) / bg_std if bg_std > 0 else 0
            
            # Count as detection if SNR is high enough and position error is small
            if snr > 2.0 and pos_error < search_radius * 0.8:
                detection_count += 1
                snr_values.append(snr)
                position_errors.append(pos_error)
    
    # Calculate final score based on multiple metrics
    if detection_count == 0:
        return 0.0
    
    # Detection fraction (how many images contain the object)
    detection_fraction = detection_count / len(original_images)
    
    # Average SNR of detections
    avg_snr = np.mean(snr_values) if snr_values else 0
    snr_score = min(1.0, avg_snr / 10.0)  # Cap at SNR=10
    
    # Position consistency (lower is better)
    avg_pos_error = np.mean(position_errors) if position_errors else search_radius
    pos_score = 1.0 - (avg_pos_error / search_radius)
    
    # Combined score with weightings
    score = (
        0.5 * detection_fraction +  # 50% weight for detection rate
        0.3 * snr_score +           # 30% weight for signal strength
        0.2 * pos_score             # 20% weight for position consistency
    )
    
    return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1