"""
mast/download.py - Download JWST FITS files from MAST

This module handles downloading FITS files from MAST based on the
search results, with support for resuming interrupted downloads and
organizing files by field ID.
"""

import os
import json
import time
import logging
import traceback
import requests
from urllib.parse import urlparse
from datetime import datetime
from tqdm import tqdm
import shutil

# Import utilities
from mast.utils import generate_timestamp, save_json, load_json

# Set up logger
logger = logging.getLogger('mast_kbo')

# MAST API token configuration (optional, for faster downloads)
# If you have a MAST API token, you can set it here
MAST_API_TOKEN = None  # Replace with your token if you have one

# Base URL for MAST direct downloads
MAST_BASE_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri="

class DownloadError(Exception):
    """Custom exception for download errors."""
    pass

def initialize_download_status(catalog_file, output_dir):
    """
    Initialize or load the download status for a catalog
    
    Parameters:
    -----------
    catalog_file : str
        Path to the catalog JSON file
    output_dir : str
        Directory where downloads will be saved
    
    Returns:
    --------
    tuple : (status_dict, status_file_path)
    """
    # Create status directory
    status_dir = os.path.join(output_dir, "status")
    os.makedirs(status_dir, exist_ok=True)
    
    # Generate or extract timestamp from catalog filename
    timestamp = None
    catalog_basename = os.path.basename(catalog_file)
    
    # Try to extract timestamp from filename
    parts = catalog_basename.split('_')
    for part in parts:
        if len(part) >= 8 and part.isdigit():
            timestamp = part
            break
    
    # If no timestamp found, generate one
    if timestamp is None:
        timestamp = generate_timestamp()
    
    status_file = os.path.join(status_dir, f"download_status_{timestamp}.json")
    
    # Check if status file already exists
    if os.path.exists(status_file):
        try:
            status = load_json(status_file)
            logger.info(f"Found existing download status: {status['downloaded']}/{status['total']} files downloaded")
            return status, status_file
        except Exception as e:
            logger.warning(f"Error loading existing status file: {e}")
            # Fall through to create a new one
    
    # Initialize new status
    status = {
        'catalog': catalog_file,
        'timestamp': timestamp,
        'start_time': datetime.now().isoformat(),
        'total': 0,
        'downloaded': 0,
        'failed': 0,
        'completed_urls': [],
        'failed_urls': [],
        'in_progress': False
    }
    
    return status, status_file

def update_download_status(status, status_file):
    """
    Update the download status file
    
    Parameters:
    -----------
    status : dict
        Download status dictionary
    status_file : str
        Path to status file
    """
    # Update timestamps
    status['last_updated'] = datetime.now().isoformat()
    
    # Save to file
    save_json(status, status_file)

def get_download_directory(output_dir, field_id):
    """
    Get the download directory for a specific field
    
    Parameters:
    -----------
    output_dir : str
        Base output directory
    field_id : str
        Field ID
    
    Returns:
    --------
    str : Full path to download directory
    """
    # Normalize field_id for use as directory name
    safe_field_id = field_id.replace('/', '_').replace('\\', '_')
    
    # Create the full download path
    download_dir = os.path.join(output_dir, "fits", safe_field_id)
    os.makedirs(download_dir, exist_ok=True)
    
    return download_dir

def process_download_url(url):
    """
    Process a MAST download URL to ensure it's in the correct format
    
    Parameters:
    -----------
    url : str
        URL or MAST URI
    
    Returns:
    --------
    str : Processed URL ready for downloading
    """
    if not url:
        raise ValueError("Empty URL provided")
    
    if url.startswith('http'):
        # URL is already in full form
        return url
    elif url.startswith('mast:'):
        # Convert MAST URI to download URL
        return f"{MAST_BASE_URL}{url}"
    else:
        # Assume it's a relative URI
        return f"{MAST_BASE_URL}mast:{url}"

def download_file(url, output_path, headers=None, chunk_size=8192, timeout=60):
    """
    Download a file with progress bar
    
    Parameters:
    -----------
    url : str
        URL to download
    output_path : str
        Output file path
    headers : dict or None
        HTTP headers for the request
    chunk_size : int
        Size of chunks to download
    timeout : int
        Request timeout in seconds
    
    Returns:
    --------
    bool : True if download successful, False otherwise
    """
    # Process URL
    try:
        download_url = process_download_url(url)
    except ValueError as e:
        logger.error(f"Invalid URL: {e}")
        return False
    
    # Set up headers
    if headers is None:
        headers = {}
    
    if MAST_API_TOKEN:
        headers['Authorization'] = f"token {MAST_API_TOKEN}"
    
    # Create temporary file for download
    temp_path = f"{output_path}.part"
    
    try:
        # Check if file already exists at destination
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            return True
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Start download
        response = requests.get(download_url, headers=headers, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Get file size for progress bar
        file_size = int(response.headers.get('content-length', 0))
        
        # Progress bar for download
        with open(temp_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    bar.update(len(chunk))
        
        # Move the temporary file to the final destination
        shutil.move(temp_path, output_path)
        
        return True
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Download error for {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {url}: {e}")
        traceback.print_exc()
        return False
    finally:
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def download_observation(observation, field_id, output_dir, status):
    """
    Download a single observation
    
    Parameters:
    -----------
    observation : dict
        Observation dictionary from catalog
    field_id : str
        Field ID for organizing downloads
    output_dir : str
        Base output directory
    status : dict
        Download status dictionary
    
    Returns:
    --------
    bool : True if download successful, False otherwise
    """
    # Extract download URL
    download_url = observation.get('dataURL')
    if not download_url:
        logger.warning(f"No download URL for observation {observation.get('product_id', 'unknown')}")
        return False
    
    # Check if already downloaded
    if download_url in status['completed_urls']:
        logger.info(f"Already downloaded: {download_url}")
        return True
    
    # Get the filename from the URL
    try:
        parsed_url = urlparse(download_url)
        filename = os.path.basename(parsed_url.path)
        
        # If no filename could be extracted, use the product ID or a generated name
        if not filename:
            filename = observation.get('product_id', f"product_{len(status['completed_urls'])}.fits")
            
        # Ensure filename ends with .fits
        if not filename.lower().endswith('.fits'):
            filename += '.fits'
    except:
        # Fallback filename
        filename = f"observation_{len(status['completed_urls'])}.fits"
    
    # Get download directory
    download_dir = get_download_directory(output_dir, field_id)
    output_path = os.path.join(download_dir, filename)
    
    # Log the download attempt
    logger.info(f"Downloading {filename} to {download_dir}")
    
    # Perform the download
    success = download_file(download_url, output_path)
    
    if success:
        logger.info(f"Successfully downloaded {filename}")
        status['downloaded'] += 1
        status['completed_urls'].append(download_url)
        return True
    else:
        logger.error(f"Failed to download {filename}")
        status['failed'] += 1
        status['failed_urls'].append(download_url)
        return False

def process_sequence(sequence, output_dir, status, delay=1.0):
    """
    Process a sequence and download its observations
    
    Parameters:
    -----------
    sequence : dict
        Sequence dictionary from catalog
    output_dir : str
        Base output directory
    status : dict
        Download status dictionary
    delay : float
        Delay between downloads in seconds
    
    Returns:
    --------
    int : Number of successful downloads
    """
    field_id = sequence.get('field_id', 'unknown')
    observations = sequence.get('observations', [])
    
    logger.info(f"Processing sequence for field {field_id}: {len(observations)} observations")
    
    # Log sequence information if available
    if all(key in sequence for key in ['center_ra', 'center_dec']):
        logger.info(f"Field center: RA={sequence['center_ra']:.6f}, Dec={sequence['center_dec']:.6f}")
    
    if all(key in sequence for key in ['start_time', 'end_time', 'duration_hours']):
        logger.info(f"Observation period: {sequence['start_time']} to {sequence['end_time']} "
                   f"({sequence.get('duration_hours', 0):.2f} hours)")
    
    # Download each observation in the sequence
    successful_downloads = 0
    
    for i, obs in enumerate(observations):
        logger.info(f"Downloading observation {i+1}/{len(observations)}")
        
        if download_observation(obs, field_id, output_dir, status):
            successful_downloads += 1
        
        # Update status after each download
        update_download_status(status, status_file)
        
        # Pause briefly between downloads
        if i < len(observations) - 1:
            time.sleep(delay)
    
    return successful_downloads

def download_from_catalog(catalog_file, output_dir, resume=False, delay=1.0):
    """
    Download FITS files from a catalog
    
    Parameters:
    -----------
    catalog_file : str
        Path to catalog JSON file
    output_dir : str
        Base output directory
    resume : bool
        Whether to resume from previous state
    delay : float
        Delay between downloads in seconds
    
    Returns:
    --------
    dict : Download status
    """
    global status_file
    
    # Load catalog
    try:
        catalog = load_json(catalog_file)
        if isinstance(catalog, list):
            # List of sequences
            sequences = catalog
        elif isinstance(catalog, dict) and 'results' in catalog:
            # Combined results format
            sequences = []
            for result in catalog['results']:
                if 'observations' in result and result['observations']:
                    sequences.append({
                        'field_id': result.get('square_id', 'unknown'),
                        'center_ra': result.get('center_ra'),
                        'center_dec': result.get('center_dec'),
                        'observations': result['observations']
                    })
        else:
            raise ValueError("Unknown catalog format")
        
        logger.info(f"Loaded catalog with {len(sequences)} sequences")
    except Exception as e:
        logger.error(f"Error loading catalog: {e}")
        return None
    
    # Initialize or load download status
    status, status_file = initialize_download_status(catalog_file, output_dir)
    
    # Count total files if starting fresh
    if not resume or status['total'] == 0:
        status['total'] = sum(len(seq.get('observations', [])) for seq in sequences)
    
    # Mark as in progress
    status['in_progress'] = True
    update_download_status(status, status_file)
    
    # Create main output directory structure
    fits_dir = os.path.join(output_dir, "fits")
    os.makedirs(fits_dir, exist_ok=True)
    
    # Process each sequence
    try:
        for i, sequence in enumerate(sequences):
            logger.info(f"\n=== Sequence {i+1}/{len(sequences)} ===")
            process_sequence(sequence, output_dir, status, delay)
    
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user. Progress has been saved.")
    
    except Exception as e:
        logger.error(f"Error during download: {e}")
        traceback.print_exc()
    
    finally:
        # Mark as no longer in progress
        status['in_progress'] = False
        status['end_time'] = datetime.now().isoformat()
        update_download_status(status, status_file)
        
        # Print summary
        logger.info("\n=== Download Summary ===")
        logger.info(f"Total files: {status['total']}")
        logger.info(f"Downloaded: {status['downloaded']}")
        logger.info(f"Failed: {status['failed']}")
        logger.info(f"Remaining: {status['total'] - status['downloaded']}")
        
        if status['failed_urls']:
            logger.info("\nFailed downloads:")
            for url in status['failed_urls'][:10]:  # Show only first 10
                logger.info(f"  - {url}")
            
            if len(status['failed_urls']) > 10:
                logger.info(f"  ... and {len(status['failed_urls'])-10} more")
            
            logger.info("\nRun with --resume flag to retry failed downloads")
    
    return status

def retry_failed_downloads(status_file_path, output_dir, delay=1.0):
    """
    Retry failed downloads from a status file
    
    Parameters:
    -----------
    status_file_path : str
        Path to download status file
    output_dir : str
        Base output directory
    delay : float
        Delay between downloads in seconds
    
    Returns:
    --------
    dict : Updated download status
    """
    global status_file
    status_file = status_file_path
    
    # Load status file
    try:
        status = load_json(status_file_path)
        if not status or 'failed_urls' not in status:
            logger.error("Invalid status file or no failed downloads to retry")
            return None
        
        failed_urls = status['failed_urls']
        logger.info(f"Found {len(failed_urls)} failed downloads to retry")
        
        if not failed_urls:
            logger.info("No failed downloads to retry")
            return status
    except Exception as e:
        logger.error(f"Error loading status file: {e}")
        return None
    
    # Create a list of failed downloads to retry
    retry_items = []
    catalog_file = status.get('catalog')
    
    if catalog_file and os.path.exists(catalog_file):
        # Try to get original observations from catalog
        try:
            catalog = load_json(catalog_file)
            
            # Extract all observations from catalog
            all_observations = []
            
            if isinstance(catalog, list):
                # List of sequences
                for seq in catalog:
                    if 'observations' in seq:
                        field_id = seq.get('field_id', 'unknown')
                        for obs in seq['observations']:
                            obs['field_id'] = field_id
                            all_observations.append(obs)
            elif isinstance(catalog, dict) and 'results' in catalog:
                # Combined results format
                for result in catalog['results']:
                    if 'observations' in result:
                        field_id = result.get('square_id', 'unknown')
                        for obs in result['observations']:
                            obs['field_id'] = field_id
                            all_observations.append(obs)
            
            # Match failed URLs to observations
            for url in failed_urls:
                for obs in all_observations:
                    if obs.get('dataURL') == url:
                        retry_items.append({
                            'url': url,
                            'field_id': obs.get('field_id', 'unknown'),
                            'observation': obs
                        })
                        break
                else:
                    # If no matching observation found, create a minimal one
                    retry_items.append({
                        'url': url,
                        'field_id': 'unknown',
                        'observation': {'dataURL': url}
                    })
            
        except Exception as e:
            logger.warning(f"Error matching failed URLs to catalog: {e}")
            # Fall back to minimal retry items
            retry_items = [{'url': url, 'field_id': 'unknown', 'observation': {'dataURL': url}} 
                         for url in failed_urls]
    else:
        # Without catalog, create minimal retry items
        retry_items = [{'url': url, 'field_id': 'unknown', 'observation': {'dataURL': url}} 
                     for url in failed_urls]
    
    # Mark as in progress
    status['in_progress'] = True
    update_download_status(status, status_file_path)
    
    # Retry each download
    successful_retries = 0
    new_failures = []
    
    try:
        for i, item in enumerate(retry_items):
            logger.info(f"Retrying {i+1}/{len(retry_items)}: {item['url']}")
            
            if download_observation(item['observation'], item['field_id'], output_dir, status):
                successful_retries += 1
                # Remove from failed_urls list
                if item['url'] in status['failed_urls']:
                    status['failed_urls'].remove(item['url'])
            else:
                new_failures.append(item['url'])
            
            # Update status after each download attempt
            update_download_status(status, status_file_path)
            
            # Pause briefly between downloads
            if i < len(retry_items) - 1:
                time.sleep(delay)
    
    except KeyboardInterrupt:
        logger.warning("\nRetry interrupted by user. Progress has been saved.")
    
    except Exception as e:
        logger.error(f"Error during retry: {e}")
        traceback.print_exc()
    
    finally:
        # Mark as no longer in progress
        status['in_progress'] = False
        status['end_time'] = datetime.now().isoformat()
        update_download_status(status, status_file_path)
        
        # Print summary
        logger.info("\n=== Retry Summary ===")
        logger.info(f"Total retried: {len(retry_items)}")
        logger.info(f"Successful: {successful_retries}")
        logger.info(f"Failed again: {len(new_failures)}")
        
        # Update overall status
        status['failed'] = len(status['failed_urls'])
        
        logger.info("\n=== Overall Status ===")
        logger.info(f"Total files: {status['total']}")
        logger.info(f"Downloaded: {status['downloaded']}")
        logger.info(f"Failed: {status['failed']}")
        logger.info(f"Remaining: {status['total'] - status['downloaded']}")
    
    return status