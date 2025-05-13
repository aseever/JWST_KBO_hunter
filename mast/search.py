"""
mast/search.py - JWST observation search functionality

This module handles searching the MAST archive for JWST observations
with focused filtering for KBO detection.
"""

import os
import time
import threading
import json
from datetime import datetime
import numpy as np
import logging
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from astroquery.mast import Observations
from astroquery.exceptions import TimeoutError as AstroQueryTimeoutError

# Import utilities
from mast.utils import (
    generate_timestamp, 
    is_near_ecliptic, 
    KBO_DETECTION_CONSTANTS,
    update_square_status
)

# Set up logger
logger = logging.getLogger('mast_kbo')

class MastSearchTimeoutError(Exception):
    """Custom exception for MAST search timeouts."""
    pass

def search_mast_with_timeout(coords, timeout=300, max_retries=3, retry_delay=30):
    """
    Search MAST with a timeout, using threading.Timer for Windows compatibility
    
    Parameters:
    -----------
    coords : dict
        Dictionary with ra_min, ra_max, dec_min, dec_max or center_ra, center_dec
    timeout : int
        Timeout in seconds (default: 5 minutes)
    max_retries : int
        Maximum number of retry attempts
    retry_delay : int
        Delay between retries in seconds
    
    Returns:
    --------
    astropy.table.Table or None : Search results from MAST
    """
    # Calculate center and radius for search
    if all(k in coords for k in ['ra_min', 'ra_max', 'dec_min', 'dec_max']):
        # Calculate center from bounds
        ra_center = (coords['ra_min'] + coords['ra_max']) / 2
        dec_center = (coords['dec_min'] + coords['dec_max']) / 2
        
        # Calculate the radius to encompass our box (using the diagonal)
        ra_span = abs(coords['ra_max'] - coords['ra_min'])
        dec_span = abs(coords['dec_max'] - coords['dec_min'])
        radius = (ra_span**2 + dec_span**2)**0.5 / 2  # Half of the diagonal
    elif all(k in coords for k in ['center_ra', 'center_dec', 'radius']):
        # Use provided center and radius
        ra_center = coords['center_ra']
        dec_center = coords['center_dec']
        radius = coords['radius']
    else:
        raise ValueError("Coordinates must contain either bounds (ra_min, ra_max, dec_min, dec_max) or center and radius (center_ra, center_dec, radius)")
    
    center_coords = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.deg, u.deg))
    logger.info(f"Searching MAST at RA={ra_center:.6f}, Dec={dec_center:.6f} with radius={radius:.6f}°")
    
    # Set up thread-safe result container
    result = {"observations": None, "completed": False, "error": None}
    
    # Counter for retry attempts
    attempts = 0
    
    try:
        while attempts < max_retries:
            attempts += 1
            
            if attempts > 1:
                logger.info(f"Retry attempt {attempts}/{max_retries} after {retry_delay}s delay")
                time.sleep(retry_delay)
            
            # Reset result for this attempt
            result["observations"] = None
            result["completed"] = False
            result["error"] = None
            
            # Search function to run in a separate thread
            def search_function():
                try:
                    # Set the timeout for the Astroquery MAST service
                    Observations.TIMEOUT = timeout
                    
                    # Search using query_region WITHOUT filtering parameters
                    observations = Observations.query_region(
                        coordinates=center_coords, 
                        radius=radius * u.deg
                    )
                    
                    # Filter for JWST observations AFTER the search
                    if observations is not None and len(observations) > 0:
                        # Filter for JWST
                        observations = observations[observations['obs_collection'] == 'JWST']
                        
                        # Early filtering for MIRI instrument
                        if 'instrument_name' in observations.colnames:
                            miri_mask = np.array(['MIRI' in str(inst) for inst in observations['instrument_name']])
                            if np.any(miri_mask):
                                observations = observations[miri_mask]
                        
                        # Filter for science observations (not calibration)
                        if 'intentType' in observations.colnames:
                            science_mask = observations['intentType'] == 'science'
                            if np.any(science_mask):
                                observations = observations[science_mask]
                        
                        # Filter for minimum exposure time
                        if 't_exptime' in observations.colnames:
                            min_exp = KBO_DETECTION_CONSTANTS['MIN_EXPOSURE_TIME']
                            exp_mask = observations['t_exptime'] >= min_exp
                            if np.any(exp_mask):
                                observations = observations[exp_mask]
                    
                    result["observations"] = observations
                    result["completed"] = True
                except Exception as e:
                    result["error"] = str(e)
                    result["completed"] = True
            
            # Create and start search thread
            search_thread = threading.Thread(target=search_function)
            search_thread.daemon = True
            
            # Track start time for timeout
            start_time = time.time()
            search_thread.start()
            
            # Wait for thread to complete or timeout
            try:
                while time.time() - start_time < timeout:
                    if result["completed"]:
                        if result["error"]:
                            logger.warning(f"Error during MAST search: {result['error']}")
                            break
                        return result["observations"]
                    
                    # Show progress indicator every 15 seconds
                    elapsed_time = time.time() - start_time
                    if int(elapsed_time) % 15 == 0:
                        minutes = int(elapsed_time) // 60
                        seconds = int(elapsed_time) % 60
                        logger.debug(f"Search in progress: {minutes}m {seconds}s elapsed")
                    
                    time.sleep(0.5)  # Check every half second
            except KeyboardInterrupt:
                logger.warning("Search interrupted by user")
                # Re-raise instead of returning None
                raise
            
            # If we're here and completed is False, we've timed out
            if not result["completed"]:
                logger.warning(f"Search timed out after {timeout} seconds")
                # Continue to retry
            elif result["observations"] is not None:
                # Success!
                return result["observations"]
        
        # If we've exhausted all retries
        logger.error(f"Failed to complete MAST search after {max_retries} attempts")
        raise MastSearchTimeoutError(f"MAST search timed out after {max_retries} attempts")
    
    except KeyboardInterrupt:
        logger.critical("Keyboard interrupt received, terminating search")
        raise

def convert_table_to_dict(table):
    """
    Convert an Astropy Table to a list of dictionaries for JSON serialization
    
    Parameters:
    -----------
    table : astropy.table.Table or list
        Table or list to convert
        
    Returns:
    --------
    list : List of dictionaries
    """
    result_list = []
    
    if hasattr(table, 'colnames'):  # It's an Astropy Table
        for row in table:
            row_dict = {}
            for col in table.colnames:
                # Convert values to JSON-serializable types
                try:
                    # Use item() to convert numpy types to Python types
                    if hasattr(row[col], 'item'):
                        row_dict[col] = row[col].item()
                    else:
                        row_dict[col] = row[col]
                except (ValueError, TypeError):
                    # If conversion fails, use string representation
                    row_dict[col] = str(row[col])
            result_list.append(row_dict)
    else:  # It's already a list of dictionaries or another iterable
        for item in table:
            if isinstance(item, dict):
                result_list.append(item)
            else:
                # Try to convert to dict if possible
                try:
                    result_list.append(dict(item))
                except (ValueError, TypeError):
                    # If conversion fails, add as is
                    result_list.append(item)
    
    return result_list

def get_product_list(observations, filter_level=2, only_science=True):
    """
    Get FITS products from observations with targeted filtering
    
    Parameters:
    -----------
    observations : astropy.table.Table
        Table of observations from MAST search
    filter_level : int
        Minimum calibration level (1=raw, 2=calibrated, 3=science products)
    only_science : bool
        Whether to only include science products
    
    Returns:
    --------
    list : Filtered list of products
    """
    if observations is None or len(observations) == 0:
        logger.info("No observations to get products for")
        return []
    
    logger.info(f"Getting products for {len(observations)} observations...")
    
    try:
        # Get data products
        products = Observations.get_product_list(observations)
        num_total = len(products)
        logger.info(f"Found {num_total} total products")
        
        # Apply focused filtering for KBO detection
        # 1. Only FITS files
        fits_mask = np.array([str(f).lower().endswith('.fits') for f in products['productFilename']])
        products = products[fits_mask]
        logger.info(f"After FITS filter: {len(products)}/{num_total} products")
        
        # 2. Only science products (if requested)
        if only_science and 'productSubGroupDescription' in products.colnames:
            science_mask = products['productSubGroupDescription'] == 'SCIENCE'
            products = products[science_mask]
            logger.info(f"After science filter: {len(products)}/{num_total} products")
        
        # 3. Minimum calibration level
        if 'calib_level' in products.colnames:
            calib_mask = products['calib_level'] >= filter_level
            products = products[calib_mask]
            logger.info(f"After calibration level filter: {len(products)}/{num_total} products")
        
        # 4. Filter by MIRI filters if available
        if 'filters' in products.colnames:
            preferred = KBO_DETECTION_CONSTANTS['PREFERRED_FILTERS']
            filter_mask = np.zeros(len(products), dtype=bool)
            
            for i, filter_val in enumerate(products['filters']):
                filter_str = str(filter_val)
                if any(pref in filter_str for pref in preferred):
                    filter_mask[i] = True
            
            if np.any(filter_mask):
                products = products[filter_mask]
                logger.info(f"After MIRI filter filter: {len(products)}/{num_total} products")
        
        # Convert to dictionaries for easier handling
        product_dicts = convert_table_to_dict(products)
        logger.info(f"Returning {len(product_dicts)} filtered products")
        
        return product_dicts
    
    except Exception as e:
        logger.error(f"Error getting product list: {e}")
        import traceback
        traceback.print_exc()
        return []

def find_observation_sequences(products, min_interval_hours=2.0, max_interval_hours=24.0):
    """
    Find observation sequences suitable for KBO detection
    
    Parameters:
    -----------
    products : list
        List of product dictionaries
    min_interval_hours : float
        Minimum time between observations in hours
    max_interval_hours : float
        Maximum time between observations in hours
    
    Returns:
    --------
    list : List of sequence dictionaries
    """
    if not products:
        return []
    
    # Try to find a time field
    time_fields = ['date_obs', 't_min', 'obs_time']
    time_field = None
    
    for field in time_fields:
        if field in products[0]:
            time_field = field
            break
    
    if not time_field:
        logger.warning("No time field found in products, cannot detect sequences")
        return []
    
    # Extract times and sort products by time
    products_with_time = []
    
    for product in products:
        if time_field in product:
            try:
                time_val = product[time_field]
                if isinstance(time_val, str):
                    time_obj = Time(time_val, format='isot')
                    products_with_time.append((product, time_obj.mjd))
                else:
                    # Assume it's already MJD
                    products_with_time.append((product, float(time_val)))
            except (ValueError, TypeError):
                logger.debug(f"Error parsing time value: {product[time_field]}")
    
    if not products_with_time:
        logger.warning("No products with valid time values")
        return []
    
    # Sort by time
    products_with_time.sort(key=lambda x: x[1])
    
    # Find sequences with appropriate spacing
    sequences = []
    current_sequence = [products_with_time[0]]
    
    for i in range(1, len(products_with_time)):
        curr_product, curr_time = products_with_time[i]
        prev_product, prev_time = current_sequence[-1]
        
        # Calculate time difference in hours
        time_diff_hours = (curr_time - prev_time) * 24.0
        
        if min_interval_hours <= time_diff_hours <= max_interval_hours:
            # This observation continues the sequence
            current_sequence.append((curr_product, curr_time))
        else:
            # Time gap outside our range - end current sequence and start a new one
            if len(current_sequence) >= 2:  # Only keep sequences with at least 2 observations
                seq_products = [item[0] for item in current_sequence]
                start_time = Time(current_sequence[0][1], format='mjd').iso
                end_time = Time(current_sequence[-1][1], format='mjd').iso
                duration_hours = (current_sequence[-1][1] - current_sequence[0][1]) * 24.0
                
                sequences.append({
                    'num_observations': len(seq_products),
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_hours': duration_hours,
                    'observations': seq_products
                })
            
            # Start a new sequence
            current_sequence = [(curr_product, curr_time)]
    
    # Add the last sequence if it meets criteria
    if len(current_sequence) >= 2:
        seq_products = [item[0] for item in current_sequence]
        start_time = Time(current_sequence[0][1], format='mjd').iso
        end_time = Time(current_sequence[-1][1], format='mjd').iso
        duration_hours = (current_sequence[-1][1] - current_sequence[0][1]) * 24.0
        
        sequences.append({
            'num_observations': len(seq_products),
            'start_time': start_time,
            'end_time': end_time,
            'duration_hours': duration_hours,
            'observations': seq_products
        })
    
    # Sort sequences by number of observations (descending)
    sequences.sort(key=lambda x: x['num_observations'], reverse=True)
    
    logger.info(f"Found {len(sequences)} observation sequences")
    return sequences

def process_square(square, status_dict, output_dir=None, timeout=300):
    """
    Process a single search square with focused KBO detection criteria
    
    Parameters:
    -----------
    square : dict
        Square dictionary with coordinate information
    status_dict : dict
        Status dictionary for tracking progress
    output_dir : str or None
        Output directory for results
    timeout : int
        Search timeout in seconds
    
    Returns:
    --------
    dict : Results including observation sequences
    """
    square_id = square['square_id']
    logger.info(f"Processing square {square_id}")
    
    # Update status
    update_square_status(status_dict, square_id, 
                        processed=False, 
                        start_time=datetime.now().isoformat())
    
    try:
        # Search MAST with early filtering
        observations = search_mast_with_timeout(square, timeout=timeout)
        
        if observations is None or len(observations) == 0:
            logger.info(f"No observations found in square {square_id}")
            update_square_status(status_dict, square_id, 
                                processed=True, 
                                end_time=datetime.now().isoformat(),
                                observations_found=0,
                                candidates_found=0)
            return {
                'square_id': square_id,
                'center_ra': square['center_ra'],
                'center_dec': square['center_dec'],
                'total_observations': 0,
                'kbo_candidates': 0,
                'sequences': []
            }
        
        # Get KBO-filtered products
        logger.info(f"Found {len(observations)} JWST observations in square {square_id}")
        products = get_product_list(observations, filter_level=2, only_science=False)
        
        if not products:
            logger.info(f"No suitable products found in square {square_id}")
            update_square_status(status_dict, square_id, 
                                processed=True, 
                                end_time=datetime.now().isoformat(),
                                observations_found=len(observations),
                                candidates_found=0)
            return {
                'square_id': square_id,
                'center_ra': square['center_ra'],
                'center_dec': square['center_dec'],
                'total_observations': len(observations),
                'kbo_candidates': 0,
                'sequences': []
            }
        
        # Find observation sequences
        min_interval = KBO_DETECTION_CONSTANTS['MIN_SEQUENCE_INTERVAL']
        max_interval = KBO_DETECTION_CONSTANTS['MAX_SEQUENCE_INTERVAL']
        
        sequences = find_observation_sequences(
            products, 
            min_interval_hours=min_interval,
            max_interval_hours=max_interval
        )
        
        # Prepare result with only the essential information
        result = {
            'square_id': square_id,
            'center_ra': square['center_ra'],
            'center_dec': square['center_dec'],
            'total_observations': len(observations),
            'kbo_candidates': len(products),
            'sequences': sequences,
            'has_sequences': len(sequences) > 0,
            'near_ecliptic': is_near_ecliptic(square['center_ra'], square['center_dec'], 5.0)
        }
        
        # Update status
        update_square_status(status_dict, square_id, 
                            processed=True, 
                            end_time=datetime.now().isoformat(),
                            observations_found=len(observations),
                            candidates_found=len(products),
                            sequences_found=len(sequences))
        
        # Save result to file if output directory specified
        if output_dir is not None:
            timestamp = generate_timestamp()
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{square_id}_{timestamp}.json")
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            logger.info(f"Saved results to {output_file}")
        
        return result
    
    except KeyboardInterrupt:
        logger.warning(f"Processing of square {square_id} interrupted by user")
        update_square_status(status_dict, square_id, 
                            processed=False, 
                            end_time=datetime.now().isoformat(),
                            error="Interrupted by user")
        # Re-raise to propagate up the call stack
        raise
    except Exception as e:
        logger.error(f"Error processing square {square_id}: {e}")
        import traceback
        traceback.print_exc()
        update_square_status(status_dict, square_id, 
                            processed=True, 
                            end_time=datetime.now().isoformat(),
                            error=str(e))
        return {
            'square_id': square_id,
            'error': str(e)
        }

def search_multiple_squares(squares, output_dir=None, max_concurrent=1, timeout=300):
    """
    Process multiple search squares with efficient KBO detection
    
    Parameters:
    -----------
    squares : list
        List of square dictionaries
    output_dir : str or None
        Output directory for results
    max_concurrent : int
        Maximum number of concurrent searches (>1 for parallel)
    timeout : int
        Search timeout in seconds
    
    Returns:
    --------
    dict : Combined results
    """
    # Create status dictionary
    status_dict = {}
    for square in squares:
        status_dict[square['square_id']] = {
            'processed': False,
            'start_time': None,
            'end_time': None,
            'error': None,
            'observations_found': 0,
            'candidates_found': 0,
            'sequences_found': 0
        }
    
    combined_results = {
        'timestamp': datetime.now().isoformat(),
        'num_squares': len(squares),
        'results': []
    }
    
    try:
        # Process sequentially for now
        for i, square in enumerate(squares):
            logger.info(f"Processing square {i+1}/{len(squares)}: {square['square_id']}")
            result = process_square(
                square, 
                status_dict, 
                output_dir=output_dir,
                timeout=timeout
            )
            combined_results['results'].append(result)
    
    except KeyboardInterrupt:
        logger.critical("\n\nSearch interrupted by user (Ctrl+C). Terminating pipeline.")
        # Save partial results before exiting
        if output_dir is not None and combined_results['results']:
            timestamp = generate_timestamp()
            partial_file = os.path.join(output_dir, f"partial_results_{timestamp}.json")
            
            try:
                with open(partial_file, 'w') as f:
                    json.dump(combined_results, f, indent=2, default=str)
                
                logger.info(f"Saved partial results to {partial_file}")
            except Exception as e:
                logger.error(f"Error saving partial results: {e}")
        
        # Re-raise to ensure the whole process terminates
        raise
    
    # Calculate summary statistics
    total_observations = 0
    total_candidates = 0
    total_sequences = 0
    sequences_with_data = []
    
    for result in combined_results['results']:
        if 'total_observations' in result:
            total_observations += result['total_observations']
        if 'kbo_candidates' in result:
            total_candidates += result['kbo_candidates']
        if 'sequences' in result:
            total_sequences += len(result['sequences'])
            
            # Collect sequence data for final output
            if result['sequences']:
                for seq in result['sequences']:
                    seq_copy = seq.copy()
                    seq_copy['square_id'] = result['square_id']
                    seq_copy['center_ra'] = result['center_ra']
                    seq_copy['center_dec'] = result['center_dec']
                    sequences_with_data.append(seq_copy)
    
    combined_results['summary'] = {
        'total_observations': total_observations,
        'total_candidates': total_candidates,
        'total_sequences': total_sequences,
        'sequence_details': sequences_with_data
    }
    
    # Save combined results
    if output_dir is not None:
        timestamp = generate_timestamp()
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"combined_results_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        logger.info(f"Saved combined results to {output_file}")
        
        # Save a separate file just with the sequence information for easier processing
        if sequences_with_data:
            sequences_file = os.path.join(output_dir, f"sequences_{timestamp}.json")
            with open(sequences_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_sequences': len(sequences_with_data),
                    'sequences': sequences_with_data
                }, f, indent=2, default=str)
            
            logger.info(f"Saved {len(sequences_with_data)} sequences to {sequences_file}")
    
    return combined_results
    """
Fix for missing prioritize_squares function in mast.search

This is a patch to add the prioritize_squares function to mast.search.py,
which is referenced in kbo_hunt.py but was previously only implemented in
ecliptic_survey.py.
"""

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
    # Import utilities for calculating ecliptic overlap
    try:
        from mast.utils import calculate_overlap_with_ecliptic
        
        if ecliptic_priority:
            # Calculate ecliptic overlap for each square
            for square in squares:
                square['ecliptic_overlap'] = calculate_overlap_with_ecliptic(square)
            
            # Sort by ecliptic overlap (highest first)
            return sorted(squares, key=lambda s: s.get('ecliptic_overlap', 0), reverse=True)
        else:
            # Default order (as provided)
            return squares
            
    except ImportError:
        # If utils module isn't available, use a simplified approach
        if ecliptic_priority:
            # Define a simple function to estimate ecliptic proximity
            def estimate_ecliptic_proximity(square):
                # The ecliptic is roughly at dec=0, with some variation
                dec = square.get('center_dec', 0)
                if dec is None:
                    return 0
                
                # Higher score for squares closer to dec=0
                return max(0, 1.0 - (abs(dec) / 10.0))
            
            # Sort by estimated ecliptic proximity
            return sorted(squares, key=estimate_ecliptic_proximity, reverse=True)
        else:
            # Default order (as provided)
            return squares