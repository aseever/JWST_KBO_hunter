"""
mast/utils.py - Utility functions for MAST search and KBO detection

This module provides utility functions for handling coordinates, 
dividing search regions, logging, and other common operations used
throughout the MAST search and KBO detection pipeline.
"""

import os
import json
import logging
from datetime import datetime
import math
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u

# Set up logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Set up logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger('mast_kbo')

def read_coordinates(config_file):
    """
    Read coordinate box from config file and convert to decimal degrees
    
    Parameters:
    -----------
    config_file : str
        Path to config file with RA_MIN, RA_MAX, DEC_MIN, DEC_MAX
        
    Returns:
    --------
    dict : Dictionary with ra_min, ra_max, dec_min, dec_max in decimal degrees
    """
    coords = {}
    
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = [item.strip() for item in line.split('=', 1)]
                    coords[key] = value
        
        # Check if all required keys are present
        required_keys = ['RA_MIN', 'RA_MAX', 'DEC_MIN', 'DEC_MAX']
        for key in required_keys:
            if key not in coords:
                raise ValueError(f"Required key '{key}' not found in config file")
        
        # Convert coordinates to decimal degrees using Angle
        ra_min = Angle(coords['RA_MIN'], unit=u.hourangle).deg
        ra_max = Angle(coords['RA_MAX'], unit=u.hourangle).deg
        dec_min = Angle(coords['DEC_MIN'], unit=u.deg).deg
        dec_max = Angle(coords['DEC_MAX'], unit=u.deg).deg
        
        return {
            'ra_min': ra_min,
            'ra_max': ra_max,
            'dec_min': dec_min,
            'dec_max': dec_max
        }
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_file}")
    except Exception as e:
        raise ValueError(f"Error parsing coordinates: {e}")

def format_coordinates(coords):
    """
    Format coordinates as readable strings
    
    Parameters:
    -----------
    coords : dict
        Dictionary with ra_min, ra_max, dec_min, dec_max in decimal degrees
        
    Returns:
    --------
    dict : Dictionary with formatted coordinate strings
    """
    ra_min_hms = Angle(coords['ra_min'], unit=u.deg).to_string(unit=u.hourangle, sep=':', precision=1)
    ra_max_hms = Angle(coords['ra_max'], unit=u.deg).to_string(unit=u.hourangle, sep=':', precision=1)
    dec_min_dms = Angle(coords['dec_min'], unit=u.deg).to_string(unit=u.deg, sep=':', precision=1)
    dec_max_dms = Angle(coords['dec_max'], unit=u.deg).to_string(unit=u.deg, sep=':', precision=1)
    
    return {
        'ra_min_hms': ra_min_hms,
        'ra_max_hms': ra_max_hms,
        'dec_min_dms': dec_min_dms,
        'dec_max_dms': dec_max_dms
    }

def divide_region_into_squares(coords, size_deg=1.0):
    """
    Divide a region into 1-degree (or specified size) squares
    
    Parameters:
    -----------
    coords : dict
        Dictionary with ra_min, ra_max, dec_min, dec_max in decimal degrees
    size_deg : float
        Size of each square in degrees (default: 1.0)
    
    Returns:
    --------
    list : List of dictionaries with coordinates for each square
    """
    ra_min, ra_max = coords['ra_min'], coords['ra_max']
    dec_min, dec_max = coords['dec_min'], coords['dec_max']
    
    # Handle RA wrap-around at 0/360 degrees
    if ra_max < ra_min:
        ra_max += 360.0
    
    # Calculate number of squares in each dimension
    ra_steps = math.ceil((ra_max - ra_min) / size_deg)
    dec_steps = math.ceil((dec_max - dec_min) / size_deg)
    
    # Create a unique ID for each search square
    squares = []
    
    for i in range(ra_steps):
        for j in range(dec_steps):
            square_ra_min = ra_min + i * size_deg
            square_ra_max = min(ra_min + (i + 1) * size_deg, ra_max)
            square_dec_min = dec_min + j * size_deg
            square_dec_max = min(dec_min + (j + 1) * size_deg, dec_max)
            
            # Keep RA within 0-360 range
            if square_ra_min >= 360.0:
                square_ra_min -= 360.0
            if square_ra_max >= 360.0:
                square_ra_max -= 360.0
            
            square = {
                'ra_min': square_ra_min,
                'ra_max': square_ra_max,
                'dec_min': square_dec_min,
                'dec_max': square_dec_max,
                'square_id': f"RA{square_ra_min:.2f}-{square_ra_max:.2f}_DEC{square_dec_min:.2f}-{square_dec_max:.2f}"
            }
            
            # Add center coordinates for this square (useful for searches)
            square['center_ra'] = (square_ra_min + square_ra_max) / 2
            square['center_dec'] = (square_dec_min + square_dec_max) / 2
            
            # Calculate area in square degrees
            square['area_sq_deg'] = (square_ra_max - square_ra_min) * (square_dec_max - square_dec_min) * math.cos(math.radians(square['center_dec']))
            
            squares.append(square)
    
    return squares

def is_near_ecliptic(ra, dec, max_ecliptic_latitude=5.0):
    """
    Check if coordinates are close to the ecliptic plane
    
    Parameters:
    -----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    max_ecliptic_latitude : float
        Maximum allowed distance from ecliptic plane in degrees
        
    Returns:
    --------
    bool : True if coordinates are within max_ecliptic_latitude of ecliptic
    """
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    ecliptic_coord = coord.transform_to('ecliptic')
    return abs(ecliptic_coord.lat.deg) <= max_ecliptic_latitude

def save_json(data, filename, indent=2):
    """
    Save data to JSON file with proper directory creation
    
    Parameters:
    -----------
    data : dict or list
        Data to save
    filename : str
        Output filename
    indent : int
        JSON indentation level
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

def load_json(filename):
    """
    Load data from JSON file with error handling
    
    Parameters:
    -----------
    filename : str
        Input filename
        
    Returns:
    --------
    dict or list : Loaded data
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filename}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in file: {filename}")

def generate_timestamp():
    """Generate a timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_square_status_tracker(squares):
    """
    Create a dictionary to track search status for each square
    
    Parameters:
    -----------
    squares : list
        List of square dictionaries
        
    Returns:
    --------
    dict : Dictionary mapping square_id to status information
    """
    status = {}
    
    for square in squares:
        status[square['square_id']] = {
            'processed': False,
            'start_time': None,
            'end_time': None,
            'error': None,
            'observations_found': 0,
            'candidates_found': 0
        }
    
    return status

def update_square_status(status_dict, square_id, **kwargs):
    """
    Update status for a specific square
    
    Parameters:
    -----------
    status_dict : dict
        Status dictionary from create_square_status_tracker
    square_id : str
        ID of the square to update
    **kwargs : 
        Key-value pairs to update
    """
    if square_id in status_dict:
        for key, value in kwargs.items():
            status_dict[square_id][key] = value

def calculate_overlap_with_ecliptic(square, ecliptic_width=10.0):
    """
    Calculate approximate overlap of a square with the ecliptic band
    
    Parameters:
    -----------
    square : dict
        Square dictionary with ra_min, ra_max, dec_min, dec_max
    ecliptic_width : float
        Width of ecliptic band in degrees (±ecliptic_width/2)
        
    Returns:
    --------
    float : Approximate overlap percentage (0.0-1.0)
    """
    # This is a simplified calculation - we sample points in the square
    # and check how many fall within the ecliptic band
    
    samples = 100
    points_in_ecliptic = 0
    
    for i in range(samples):
        # Generate random point in square
        ra = square['ra_min'] + (square['ra_max'] - square['ra_min']) * (i % 10) / 10
        dec = square['dec_min'] + (square['dec_max'] - square['dec_min']) * (i // 10) / 10
        
        if is_near_ecliptic(ra, dec, max_ecliptic_latitude=ecliptic_width/2):
            points_in_ecliptic += 1
    
    return points_in_ecliptic / samples

def prioritize_squares(squares, ecliptic_priority=True):
    """
    Prioritize squares for searching based on criteria
    
    Parameters:
    -----------
    squares : list
        List of square dictionaries
    ecliptic_priority : bool
        Whether to prioritize squares near the ecliptic plane
        
    Returns:
    --------
    list : List of squares sorted by priority (highest first)
    """
    if ecliptic_priority:
        # Calculate ecliptic overlap for each square
        for square in squares:
            square['ecliptic_overlap'] = calculate_overlap_with_ecliptic(square)
        
        # Sort by ecliptic overlap (highest first)
        return sorted(squares, key=lambda s: s['ecliptic_overlap'], reverse=True)
    else:
        # Default order (as provided)
        return squares

# Constants for KBO detection
KBO_DETECTION_CONSTANTS = {
    # Wavelength range for optimal KBO detection (microns)
    'MIN_WAVELENGTH': 10.0,
    'MAX_WAVELENGTH': 25.0,
    
    # Preferred JWST filters for KBO detection
    'PREFERRED_FILTERS': [
        'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W'
    ],
    
    # Minimum exposure time for KBO detection (seconds)
    'MIN_EXPOSURE_TIME': 300,
    
    # Time constraints for KBO sequences (hours)
    'MIN_SEQUENCE_INTERVAL': 2.0,
    'MAX_SEQUENCE_INTERVAL': 24.0,
    
    # Motion parameters for KBOs
    'TYPICAL_MOTION_RATE': 3.0,  # arcsec/hour
    
    # Minimum field matching radius (arcsec)
    'FIELD_MATCH_RADIUS': 60.0
}