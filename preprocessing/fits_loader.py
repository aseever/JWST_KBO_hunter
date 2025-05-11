"""
FITS file loading and validation for JWST data preprocessing.

This module handles loading FITS files, particularly from JWST/MIRI, and
provides utilities for extracting and validating the data.
"""

import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

def load_fits_file(filepath, verbose=True, wavelength_idx=None):
    """
    Load a FITS file and extract primary data and header
    
    Parameters:
    -----------
    filepath : str
        Path to the FITS file
    verbose : bool
        Whether to print verbose information
    wavelength_idx : int or None
        For 3D data cubes, specify which wavelength slice to extract.
        If None, will use the middle slice.
        
    Returns:
    --------
    dict or None
        Dictionary containing the data and metadata, or None if loading failed
    """
    if verbose:
        print(f"Loading FITS file: {os.path.basename(filepath)}")
    
    try:
        with fits.open(filepath) as hdul:
            # JWST FITS files can be complex - we need to determine the right extension
            # Level 3 MIRI IFU data is typically a 3D data cube
            
            if verbose:
                print("Available FITS extensions:")
                for i, ext in enumerate(hdul):
                    print(f"  [{i}] {ext.name}: {ext.header.get('EXTNAME', 'N/A')} - "
                          f"Shape: {ext.data.shape if hasattr(ext, 'data') and ext.data is not None else 'No data'}")
            
            # Try to find the science data extension
            sci_ext = None
            for i, ext in enumerate(hdul):
                if hasattr(ext, 'data') and ext.data is not None:
                    if ext.header.get('EXTNAME') == 'SCI' or 'SCI' in ext.name:
                        sci_ext = i
                        break
            
            # If no science extension found, use primary if it has data
            if sci_ext is None:
                if hdul[0].data is not None:
                    sci_ext = 0
                else:
                    # Otherwise, use the first extension with data
                    for i, ext in enumerate(hdul):
                        if hasattr(ext, 'data') and ext.data is not None:
                            sci_ext = i
                            break
            
            if sci_ext is None:
                raise ValueError("No valid data extension found in FITS file")
            
            # Extract the data and header
            data = hdul[sci_ext].data.copy()  # Make a copy to avoid issues with closed file
            header = hdul[sci_ext].header.copy()
            primary_header = hdul[0].header.copy()  # Primary header often has important metadata
            
            # For MIRI IFU data (3D cubes), we need to extract a 2D image
            if len(data.shape) == 3:
                if verbose:
                    print(f"  3D data cube found: {data.shape}")
                
                # Determine which slice to extract
                n_slices = data.shape[0]
                if wavelength_idx is None:
                    # Get central wavelength slice by default
                    wavelength_idx = n_slices // 2
                else:
                    # Validate the provided index
                    if wavelength_idx < 0 or wavelength_idx >= n_slices:
                        print(f"  Warning: Requested slice {wavelength_idx} is out of bounds. "
                              f"Using central slice instead.")
                        wavelength_idx = n_slices // 2
                
                # Extract the slice
                data = data[wavelength_idx, :, :]
                
                if verbose:
                    print(f"  Extracted 2D slice at wavelength index {wavelength_idx}: {data.shape}")
            
            # Basic info
            if verbose:
                print(f"  Data shape: {data.shape}")
                print(f"  Data type: {data.dtype}")
                try:
                    min_val, max_val = np.nanmin(data), np.nanmax(data)
                    print(f"  Data range: {min_val:.3e} to {max_val:.3e}")
                except ValueError:
                    # Handle the case where all values are NaN
                    print("  Data range: All values are NaN")
            
            # Extract key metadata
            instrument = primary_header.get('INSTRUME', 'Unknown')
            filter_name = primary_header.get('FILTER', 'Unknown')
            exp_time = primary_header.get('EFFEXPTM', primary_header.get('EXPTIME', 0))
            
            if verbose:
                print(f"  Instrument: {instrument}")
                print(f"  Filter: {filter_name}")
                print(f"  Exposure time: {exp_time} seconds")
            
            # Try to extract WCS info (may fail for complex data structures)
            wcs = None
            try:
                wcs = WCS(header)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not extract WCS: {e}")
            
            return {
                'data': data,
                'header': header,
                'primary_header': primary_header,
                'wcs': wcs,
                'instrument': instrument,
                'filter': filter_name,
                'exptime': exp_time,
                'filename': os.path.basename(filepath)
            }
    
    except Exception as e:
        print(f"Error loading FITS file {filepath}: {e}")
        return None

def extract_wavelength_slice(data_cube, wavelength_idx=None, header=None, verbose=True):
    """
    Extract a specific wavelength slice from a 3D data cube
    
    Parameters:
    -----------
    data_cube : numpy.ndarray
        3D data cube with wavelength as the first dimension
    wavelength_idx : int or None
        Index of wavelength slice to extract. If None, uses the central slice.
    header : astropy.io.fits.Header or None
        Header containing wavelength information
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    numpy.ndarray
        2D image slice
    """
    if len(data_cube.shape) != 3:
        raise ValueError("Input data must be a 3D cube")
    
    n_wavelengths = data_cube.shape[0]
    
    # If no wavelength specified, use central slice
    if wavelength_idx is None:
        wavelength_idx = n_wavelengths // 2
    
    # Validate index
    if wavelength_idx < 0 or wavelength_idx >= n_wavelengths:
        raise ValueError(f"Wavelength index {wavelength_idx} is out of bounds [0, {n_wavelengths-1}]")
    
    # Extract slice
    slice_data = data_cube[wavelength_idx, :, :]
    
    if verbose:
        print(f"Extracted wavelength slice {wavelength_idx+1}/{n_wavelengths}")
        
        # Try to get actual wavelength value if header provided
        if header and 'WAVSTART' in header and 'WAVDELT' in header:
            wavelength = header['WAVSTART'] + wavelength_idx * header['WAVDELT']
            unit = header.get('WAVUNIT', 'μm')
            print(f"  Wavelength: {wavelength:.3f} {unit}")
    
    return slice_data

def verify_fits_structure(fits_file, verbose=True):
    """
    Verify the structure of a FITS file and print diagnostic information
    
    Parameters:
    -----------
    fits_file : str
        Path to the FITS file
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    dict
        Dictionary with verification results
    """
    results = {
        'valid': False,
        'has_data': False,
        'has_wcs': False,
        'dimensions': None,
        'instrument': None,
        'errors': []
    }
    
    try:
        with fits.open(fits_file) as hdul:
            if verbose:
                print(f"FITS file {os.path.basename(fits_file)} contains {len(hdul)} HDUs")
            
            # Check for data
            data_hdus = []
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'data') and hdu.data is not None:
                    data_hdus.append(i)
                    if verbose:
                        print(f"  HDU {i}: {hdu.name}, Shape: {hdu.data.shape}, Type: {hdu.data.dtype}")
            
            results['has_data'] = len(data_hdus) > 0
            
            if not results['has_data']:
                results['errors'].append("No data found in FITS file")
                if verbose:
                    print("  WARNING: No data found in FITS file")
            
            # Check primary header
            primary_header = hdul[0].header
            results['instrument'] = primary_header.get('INSTRUME', 'Unknown')
            
            # Try to get WCS
            for i in data_hdus:
                try:
                    wcs = WCS(hdul[i].header)
                    if wcs.has_celestial:
                        results['has_wcs'] = True
                        if verbose:
                            print(f"  HDU {i} has valid WCS")
                        break
                except Exception as e:
                    if verbose:
                        print(f"  HDU {i} WCS error: {e}")
            
            if not results['has_wcs']:
                results['errors'].append("No valid WCS found")
                if verbose:
                    print("  WARNING: No valid WCS found")
            
            # Check dimensions of first data HDU
            if data_hdus:
                results['dimensions'] = hdul[data_hdus[0]].data.shape
            
            # Overall validity
            results['valid'] = results['has_data'] and not results['errors']
            
            if verbose:
                if results['valid']:
                    print("  FITS file is valid")
                else:
                    print(f"  FITS file has issues: {', '.join(results['errors'])}")
    
    except Exception as e:
        results['errors'].append(str(e))
        if verbose:
            print(f"Error verifying FITS file: {e}")
    
    return results