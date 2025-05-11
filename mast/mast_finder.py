#!/usr/bin/env python3
"""
mast_finder.py - Script to search MAST for JWST observations in specific regions
Searches for JWST FITS files within 1°×1° squares along the ecliptic
Outputs catalog to data directory, appending after each successful square
"""

import os
import json
import datetime
import threading
import time
import sys
import math
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from astroquery.mast import Observations

# Create data directory if it doesn't exist
os.makedirs("../data", exist_ok=True)

def read_coordinates(config_file="../config/coordinates.txt"):
    """Read coordinate box from config file and convert to decimal degrees"""
    coords = {}
    
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = [item.strip() for item in line.split('=', 1)]
                coords[key] = value
    
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
    
    # Calculate number of squares in each dimension
    ra_steps = math.ceil((ra_max - ra_min) / size_deg)
    dec_steps = math.ceil((dec_max - dec_min) / size_deg)
    
    total_squares = ra_steps * dec_steps
    print(f"Dividing region into {ra_steps}×{dec_steps} = {total_squares} squares of {size_deg}°×{size_deg}°")
    
    squares = []
    
    for i in range(ra_steps):
        for j in range(dec_steps):
            square_ra_min = ra_min + i * size_deg
            square_ra_max = min(ra_min + (i + 1) * size_deg, ra_max)
            square_dec_min = dec_min + j * size_deg
            square_dec_max = min(dec_min + (j + 1) * size_deg, dec_max)
            
            square = {
                'ra_min': square_ra_min,
                'ra_max': square_ra_max,
                'dec_min': square_dec_min,
                'dec_max': square_dec_max,
                'square_id': f"RA{square_ra_min:.1f}-{square_ra_max:.1f}_DEC{square_dec_min:.1f}-{square_dec_max:.1f}"
            }
            
            squares.append(square)
    
    return squares

def search_mast_with_timeout(coords, timeout=300, wavelength_min=None, wavelength_max=None):
    """
    Search MAST with a timeout, using threading.Timer instead of signals (for Windows compatibility)
    
    Parameters:
    -----------
    coords : dict
        Dictionary with ra_min, ra_max, dec_min, dec_max in decimal degrees
    timeout : int
        Timeout in seconds (default: 5 minutes)
    wavelength_min : float
        Minimum wavelength in microns
    wavelength_max : float
        Maximum wavelength in microns
    
    Returns:
    --------
    astropy.table.Table or None : Search results from MAST
    """
    # Calculate center of our box
    ra_center = (coords['ra_min'] + coords['ra_max']) / 2
    dec_center = (coords['dec_min'] + coords['dec_max']) / 2
    
    # Calculate the radius to encompass our box (using the diagonal)
    ra_span = abs(coords['ra_max'] - coords['ra_min'])
    dec_span = abs(coords['dec_max'] - coords['dec_min'])
    radius = (ra_span**2 + dec_span**2)**0.5 / 2  # Half of the diagonal
    
    center_coords = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.deg, u.deg))
    print(f"Searching MAST centered at RA={ra_center:.6f}, Dec={dec_center:.6f} with radius={radius:.6f} degrees")
    print(f"This may take up to {timeout//60} minutes... (press Ctrl+C to abort)")
    
    # Set up threading based timeout
    result = {"observations": None, "completed": False, "error": None}
    
    def search_function():
        try:
            # Set the timeout for the Astroquery MAST service (as a fallback)
            Observations.TIMEOUT = timeout
            
            # Search using query_region
            observations = Observations.query_region(coordinates=center_coords, radius=radius * u.deg)
            
            # Filter for JWST observations
            if observations is not None and len(observations) > 0:
                observations = observations[observations['obs_collection'] == 'JWST']
            
            result["observations"] = observations
            result["completed"] = True
        except Exception as e:
            result["error"] = str(e)
    
    # Create and start search thread
    search_thread = threading.Thread(target=search_function)
    search_thread.daemon = True  # Thread will be killed when main program exits
    search_thread.start()
    
    # Wait for search to complete or timeout
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not search_thread.is_alive() or result["completed"]:
            if result["error"]:
                print(f"Error during search: {result['error']}")
                return None
            return result["observations"]
        
        # Show a progress indicator every 15 seconds
        elapsed_time = time.time() - start_time
        if int(elapsed_time) % 15 == 0:
            minutes = int(elapsed_time) // 60
            seconds = int(elapsed_time) % 60
            print(f"  Still searching... {minutes}m {seconds}s elapsed", end="\r")
        
        try:
            time.sleep(0.5)  # Check every half second
        except KeyboardInterrupt:
            print("\nSearch interrupted by user. Stopping this square...")
            return None
    
    # If we're here, we've timed out
    print(f"\nSearch timed out after {timeout} seconds. Moving to next square.")
    return None

def get_product_list(observations):
    """Get list of data products for the matching observations"""
    if observations is None or len(observations) == 0:
        return []
    
    print(f"Found {len(observations)} observations, fetching data products...")
    try:
        products = Observations.get_product_list(observations)
        
        # Filter for FITS files - using safer approaches for type checking
        science_products = []
        for row in products:
            # Check if it's a science product
            is_science = False
            if 'productSubGroupDescription' in products.colnames:
                if row['productSubGroupDescription'] == 'SCIENCE':
                    is_science = True
            
            # Check if it's a FITS file
            is_fits = False
            if 'productFilename' in products.colnames:
                filename = str(row['productFilename'])
                if filename.lower().endswith('.fits'):
                    is_fits = True
            
            if is_science and is_fits:
                science_products.append(row)
        
        return science_products
    except Exception as e:
        print(f"Error getting product list: {e}")
        print("Returning observations instead")
        return observations

def convert_table_to_dict(table):
    """Convert an Astropy Table to a list of dictionaries for JSON serialization"""
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
    else:  # It's already a list of dictionaries
        result_list = table
    
    return result_list

def append_to_json(data, filename):
    """Append data to an existing JSON file or create a new one"""
    full_path = os.path.join("../data", filename)
    
    existing_data = []
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                # If the file exists but is not valid JSON, start fresh
                existing_data = []
    
    # Convert data to list of dictionaries if it's an Astropy Table
    if data is not None:
        new_data = convert_table_to_dict(data)
        existing_data.extend(new_data)
    
    with open(full_path, 'w') as f:
        json.dump(existing_data, f, indent=2, default=str)
    
    return full_path

def main():
    # Create a timestamp for this run's output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    observations_file = f"jwst_kbo_observations_{timestamp}.json"
    summary_file = f"search_summary_{timestamp}.txt"
    
    try:
        # Read coordinates from config file
        print("Reading coordinates from config file...")
        coords = read_coordinates()
        print(f"Search box: RA [{coords['ra_min']:.6f} to {coords['ra_max']:.6f}], "
              f"Dec [{coords['dec_min']:.6f} to {coords['dec_max']:.6f}]")
        
        # Divide region into 1-degree squares
        squares = divide_region_into_squares(coords, size_deg=1.0)
        
        # Track results for summary
        square_results = []
        total_observations = 0
        
        # Open summary file
        with open(os.path.join("../data", summary_file), 'w') as summary:
            summary.write(f"JWST KBO HUNT - SEARCH SUMMARY - {datetime.datetime.now()}\n")
            summary.write(f"====================================================\n\n")
            summary.write(f"Full search region: RA [{coords['ra_min']:.6f} to {coords['ra_max']:.6f}], "
                        f"Dec [{coords['dec_min']:.6f} to {coords['dec_max']:.6f}]\n")
            summary.write(f"Divided into {len(squares)} 1°×1° squares\n\n")
            summary.write("RESULTS BY SQUARE:\n")
            summary.write("=================\n\n")
        
        # Search each square
        for i, square in enumerate(squares):
            try:
                print(f"\n=== Square {i+1}/{len(squares)}: {square['square_id']} ===")
                print(f"RA: [{square['ra_min']:.6f} to {square['ra_max']:.6f}], Dec: [{square['dec_min']:.6f} to {square['dec_max']:.6f}]")
                
                # Search MAST for JWST observations in this square
                square_results_mast = search_mast_with_timeout(square, timeout=600)
                
                if square_results_mast is None or len(square_results_mast) == 0:
                    print(f"No observations found in Square {i+1}/{len(squares)}")
                    result_info = {
                        'square_id': square['square_id'],
                        'ra_min': square['ra_min'],
                        'ra_max': square['ra_max'],
                        'dec_min': square['dec_min'],
                        'dec_max': square['dec_max'],
                        'observations': 0
                    }
                else:
                    # Add square ID to each observation for tracking
                    # Need to add a column to the table first
                    try:
                        if 'square_id' not in square_results_mast.colnames:
                            square_results_mast['square_id'] = None  # Add empty column
                        
                        # Now set values for each row
                        for row_idx in range(len(square_results_mast)):
                            square_results_mast['square_id'][row_idx] = square['square_id']
                    except Exception as e:
                        print(f"Warning: Couldn't add square_id to results: {e}")
                        # This isn't fatal, continue with the data we have
                    
                    # Append to observations file
                    print(f"Found {len(square_results_mast)} observations in Square {i+1}/{len(squares)}")
                    append_to_json(square_results_mast, observations_file)
                    total_observations += len(square_results_mast)
                    
                    result_info = {
                        'square_id': square['square_id'],
                        'ra_min': square['ra_min'],
                        'ra_max': square['ra_max'],
                        'dec_min': square['dec_min'],
                        'dec_max': square['dec_max'],
                        'observations': len(square_results_mast)
                    }
                
                square_results.append(result_info)
                
                # Update summary file
                with open(os.path.join("../data", summary_file), 'a') as summary:
                    summary.write(f"Square {i+1}/{len(squares)}: {square['square_id']}\n")
                    summary.write(f"  RA: [{square['ra_min']:.6f} to {square['ra_max']:.6f}], Dec: [{square['dec_min']:.6f} to {square['dec_max']:.6f}]\n")
                    summary.write(f"  Observations: {result_info['observations']}\n\n")
            
            except KeyboardInterrupt:
                print("\nSearch interrupted by user. Moving to next square...")
                
                # Update summary file
                with open(os.path.join("../data", summary_file), 'a') as summary:
                    summary.write(f"Square {i+1}/{len(squares)}: {square['square_id']} - INTERRUPTED\n\n")
                
                continue
        
        # Print final summary
        print("\n=== Search Complete ===")
        print(f"Searched {len(squares)} squares")
        print(f"Total observations found: {total_observations}")
        
        # Squares with observations, sorted by count
        squares_with_obs = [s for s in square_results if s['observations'] > 0]
        squares_with_obs.sort(key=lambda x: x['observations'], reverse=True)
        
        if squares_with_obs:
            print("\nTop squares with observations:")
            for i, square in enumerate(squares_with_obs[:5]):  # Show top 5
                print(f"  {i+1}. {square['square_id']}: {square['observations']} observations")
        
        # Update summary file with final summary
        with open(os.path.join("../data", summary_file), 'a') as summary:
            summary.write("\nSUMMARY:\n")
            summary.write("========\n\n")
            summary.write(f"Total observations: {total_observations}\n\n")
            
            if squares_with_obs:
                summary.write("Squares with observations (sorted by count):\n")
                for i, square in enumerate(squares_with_obs):
                    summary.write(f"{i+1}. {square['square_id']}: {square['observations']} observations\n")
        
        print(f"\nDetailed results saved to {os.path.join('../data', summary_file)}")
    
    except KeyboardInterrupt:
        print("\nSearch interrupted by user. Exiting gracefully...")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nScript execution finished.")

if __name__ == "__main__":
    main()