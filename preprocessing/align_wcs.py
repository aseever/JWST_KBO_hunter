"""
WCS-based image alignment for astronomical images.

This module provides functions to align images using World Coordinate System (WCS)
information from their FITS headers, which is particularly effective when good
astrometric calibration is available.
"""

import numpy as np
from scipy import ndimage
import time
import warnings

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
    start_time = time.time()
    
    if verbose:
        print("    Using WCS-based alignment")
    
    # Check if WCS is available
    if not ref_image_dict.get('wcs') or not target_image_dict.get('wcs'):
        if verbose:
            print("    Warning: WCS information not available")
            raise ValueError("WCS information missing")
    
    try:
        # Get WCS objects
        ref_wcs = ref_image_dict['wcs']
        target_wcs = target_image_dict['wcs']
        
        # Get reference image data
        ref_data = ref_image_dict['cleaned_data'] if 'cleaned_data' in ref_image_dict else ref_image_dict['data']
        target_data = target_image_dict['cleaned_data'] if 'cleaned_data' in target_image_dict else target_image_dict['data']
        
        # Create a grid of pixel coordinates for the reference image
        ny, nx = ref_data.shape
        
        # Use a sparser grid for very large images
        if nx * ny > 4000*4000:
            # Use a sparser grid for efficiency
            step = max(1, min(nx, ny) // 400)
            iy, ix = np.mgrid[:ny:step, :nx:step]
            if verbose:
                print(f"    Using sparse grid with step {step} for large image")
        else:
            # Use a full grid for smaller images
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
                    print(f"    WCS 3D transformation error: {e}")
                    raise ValueError("WCS transformation failed")
        else:
            try:
                # 2D WCS case
                # Convert reference pixel coordinates to sky coordinates
                ra, dec = ref_wcs.pixel_to_world(ix, iy)
                
                # Convert sky coordinates to pixel coordinates in the target image
                target_x, target_y = target_wcs.world_to_pixel(ra, dec)
            except Exception as e:
                if verbose:
                    print(f"    WCS 2D transformation error: {e}")
                    raise ValueError("WCS transformation failed")
        
        # Calculate shift using median displacement
        # This handles non-linear distortions better than a simple mean
        x_shifts = target_x - ix
        y_shifts = target_y - iy
        
        # Use robust statistics to avoid outliers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # Handle potential NaNs in the shift arrays
            valid_mask = ~np.isnan(x_shifts) & ~np.isnan(y_shifts)
            if np.sum(valid_mask) < 10:  # Need at least 10 valid points
                if verbose:
                    print("    Warning: Not enough valid WCS transformation points")
                    raise ValueError("Insufficient valid WCS points")
            
            # Sort shifts and take median
            x_shift = np.nanmedian(x_shifts[valid_mask])
            y_shift = np.nanmedian(y_shifts[valid_mask])
            
            # Calculate robust standard deviation to check consistency
            x_std = np.nanstd(x_shifts[valid_mask])
            y_std = np.nanstd(y_shifts[valid_mask])
            
            if x_std > 5.0 or y_std > 5.0:
                if verbose:
                    print(f"    Warning: High dispersion in WCS shifts (stddev x:{x_std:.2f}, y:{y_std:.2f})")
                    # Continue anyway - this warning is just informational
        
        # Check for unreasonable shifts (more than half the image size)
        max_reasonable_shift = min(nx, ny) // 2
        if abs(x_shift) > max_reasonable_shift or abs(y_shift) > max_reasonable_shift:
            if verbose:
                print(f"    Warning: Very large WCS shift detected: dx={x_shift:.1f}, dy={y_shift:.1f}")
                print(f"    Limiting to maximum reasonable shift of {max_reasonable_shift} pixels")
            
            # Limit shift to reasonable range
            x_shift = np.clip(x_shift, -max_reasonable_shift, max_reasonable_shift)
            y_shift = np.clip(y_shift, -max_reasonable_shift, max_reasonable_shift)
        
        # Apply shift to create aligned image
        aligned_data = ndimage.shift(target_data, (y_shift, x_shift), 
                                  order=1, mode='constant', cval=np.nan)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"    WCS alignment completed in {elapsed:.1f} seconds")
            print(f"    WCS alignment shift: dx={x_shift:.2f}, dy={y_shift:.2f} pixels")
        
        return aligned_data, (x_shift, y_shift)
    
    except Exception as e:
        if verbose:
            print(f"    WCS alignment failed: {e}")
        raise ValueError(f"WCS alignment error: {e}")