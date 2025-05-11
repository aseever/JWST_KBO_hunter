#!/usr/bin/env python3
"""
mast_downloader.py - Download JWST FITS files from MAST

This script reads the filtered KBO catalog created by filter_observations.py
and downloads the FITS files one by one, showing progress for each download.
Files are organized by field_id in a directory structure under data/fits/.
"""

import os
import json
import argparse
import requests
import time
from tqdm import tqdm
from urllib.parse import urlparse
from datetime import datetime

# MAST API token configuration (optional, for faster downloads)
# If you have a MAST API token, you can set it here
MAST_API_TOKEN = None  # Replace with your token if you have one

# Base URL for MAST direct downloads
MAST_BASE_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri="

def load_catalog(filename):
    """Load the filtered catalog JSON file"""
    print(f"Loading catalog from {filename}...")
    with open(filename, 'r') as f:
        catalog = json.load(f)
    
    # Check catalog format
    # Format 1: List of sequences with observations inside them
    # Format 2: Direct list of observations
    if catalog and isinstance(catalog, list):
        if isinstance(catalog[0], dict) and 'observations' in catalog[0]:
            # Format 1: List of sequences
            total_observations = sum(seq.get('num_observations', len(seq.get('observations', []))) 
                                    for seq in catalog)
            print(f"Loaded catalog with {len(catalog)} sequences containing {total_observations} observations")
            return catalog, 'sequences'
        else:
            # Format 2: Direct list of observations
            total_observations = len(catalog)
            print(f"Loaded catalog with {total_observations} direct observations")
            return catalog, 'observations'
    
    print("Warning: Empty or invalid catalog")
    return [], 'unknown'

def create_download_structure(output_dir):
    """Create the directory structure for downloads"""
    # Main FITS directory
    fits_dir = os.path.join(output_dir, "fits")
    os.makedirs(fits_dir, exist_ok=True)
    
    # Download status file directory
    os.makedirs(os.path.join(output_dir, "status"), exist_ok=True)
    
    return fits_dir

def get_download_status(output_dir, catalog_filename):
    """Get the status of downloads from previous runs"""
    # Extract timestamp from catalog filename
    catalog_basename = os.path.basename(catalog_filename)
    timestamp = catalog_basename.split('_')[-1].split('.')[0]  # Extract timestamp
    
    status_file = os.path.join(output_dir, "status", f"download_status_{timestamp}.json")
    
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            print(f"Found previous download status: {status['downloaded']}/{status['total']} files downloaded")
            return status, status_file
        except (json.JSONDecodeError, KeyError):
            # If status file is corrupted, start fresh
            pass
    
    # Initialize new status
    status = {
        'catalog': catalog_filename,
        'timestamp': timestamp,
        'total': 0,
        'downloaded': 0,
        'completed_urls': [],
        'failed_urls': []
    }
    
    return status, status_file

def update_download_status(status, status_file):
    """Update the download status file"""
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

def download_file(url, output_path, headers=None):
    """Download a file with progress bar"""
    if not url.startswith('http'):
        # Convert MAST URI to download URL
        if url.startswith('mast:'):
            url = MAST_BASE_URL + url
        else:
            raise ValueError(f"Unsupported URL format: {url}")
    
    if headers is None:
        headers = {}
    
    if MAST_API_TOKEN:
        headers['Authorization'] = f"token {MAST_API_TOKEN}"
    
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        file_size = int(response.headers.get('content-length', 0))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Progress bar for download
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    bar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def process_sequence(sequence, fits_dir, status):
    """Process a sequence and download its observations"""
    # Check if we have a sequence or a direct observation
    if 'field_id' in sequence and 'observations' in sequence:
        # This is a sequence
        field_id = sequence['field_id']
        observations = sequence['observations']
    else:
        # This is a direct observation
        field_id = sequence.get('square_id', 'unknown')
        observations = [sequence]  # Treat as a single observation
    
    field_dir = os.path.join(fits_dir, field_id)
    os.makedirs(field_dir, exist_ok=True)
    
    print(f"\nProcessing {field_id}: {len(observations)} observations")
    
    if 'center_ra' in sequence and 'center_dec' in sequence:
        print(f"Field center: RA={sequence['center_ra']:.6f}, Dec={sequence['center_dec']:.6f}")
    
    if 'start_time' in sequence and 'end_time' in sequence:
        print(f"Observation period: {sequence['start_time']} to {sequence['end_time']} ({sequence.get('duration_hours', 0):.2f} hours)")
    
    # Process each observation in the sequence
    for obs in observations:
        obs_id = obs.get('obs_id', '')
        data_url = obs.get('dataURL', '')
        
        if not data_url:
            print(f"Skipping {obs_id} - no download URL available")
            continue
        
        # Skip if already downloaded
        if data_url in status['completed_urls']:
            print(f"Skipping {obs_id} - already downloaded")
            continue
        
        # Extract filename from URL
        filename = data_url.split('/')[-1]
        output_path = os.path.join(field_dir, filename)
        
        # Show observation info
        filters = obs.get('filters', 'Unknown')
        exptime = obs.get('t_exptime', 0)
        
        print(f"\nDownloading {obs_id} ({filters}, {exptime}s exposure)")
        
        # Perform the download
        success = download_file(data_url, output_path)
        
        if success:
            print(f"Successfully downloaded {filename}")
            status['downloaded'] += 1
            status['completed_urls'].append(data_url)
        else:
            print(f"Failed to download {filename}")
            status['failed_urls'].append(data_url)
        
        # Update status after each download
        update_download_status(status, status_file)
        
        # Pause briefly between downloads
        time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Download JWST FITS files from MAST")
    parser.add_argument('catalog_file', help="Path to filtered catalog JSON file")
    parser.add_argument('--output-dir', '-o', default="../data", 
                       help="Directory to save downloaded files (default: ../data)")
    parser.add_argument('--resume', '-r', action='store_true',
                       help="Resume from previous download state")
    
    args = parser.parse_args()
    
    # Initialize
    global status_file
    catalog, catalog_type = load_catalog(args.catalog_file)
    fits_dir = create_download_structure(args.output_dir)
    status, status_file = get_download_status(args.output_dir, args.catalog_file)
    
    # Count total files if starting fresh
    if not args.resume or status['total'] == 0:
        if catalog_type == 'sequences':
            status['total'] = sum(len(seq.get('observations', [])) for seq in catalog)
        else:  # observations
            status['total'] = len(catalog)
    
    # Process each sequence/observation
    try:
        for i, item in enumerate(catalog):
            if catalog_type == 'sequences':
                print(f"\n=== Sequence {i+1}/{len(catalog)} ===")
            else:
                print(f"\n=== Observation {i+1}/{len(catalog)} ===")
            
            process_sequence(item, fits_dir, status)
    
    except KeyboardInterrupt:
        print("\nDownload interrupted by user. Progress has been saved.")
    
    finally:
        # Final status update
        update_download_status(status, status_file)
        
        # Print summary
        print("\n=== Download Summary ===")
        print(f"Total files: {status['total']}")
        print(f"Downloaded: {status['downloaded']}")
        print(f"Failed: {len(status['failed_urls'])}")
        print(f"Remaining: {status['total'] - status['downloaded']}")
        
        if len(status['failed_urls']) > 0:
            print("\nFailed downloads:")
            for url in status['failed_urls']:
                print(f"  - {url}")
            print("\nRun with --resume flag to retry failed downloads")

if __name__ == "__main__":
    main()