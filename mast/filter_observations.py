#!/usr/bin/env python3
"""
filter_observations.py - Filter JWST observations for KBO hunting

This script reads the full JWST observation catalog created by mast_finder.py
and filters it to identify the most promising observations for KBO detection.
It applies multiple filters for wavelength, exposure time, observation intent,
and looks for sequences of observations of the same field.

The output is a simplified JSON file containing only the necessary information
for the downloader script.
"""

import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

# Constants for filtering
MIN_WAVELENGTH_UM = 5.0   # Minimum wavelength in microns
MAX_WAVELENGTH_UM = 28.0  # Maximum wavelength in microns (MIRI range)
MIN_EXPOSURE_TIME = 300   # Minimum exposure time in seconds
FIELD_MATCH_RADIUS = 60   # Maximum radius in arcseconds to consider same field
KBO_MOVEMENT_RATE = 3.0   # Typical KBO movement in arcsec/hour
MIN_SEQ_INTERVAL = 0.5    # Minimum time between sequence images (hours)
MAX_SEQ_INTERVAL = 48.0   # Maximum time between sequence images (hours)
MIN_SEQUENCE_IMAGES = 2   # Minimum number of images in a sequence

# Preferred MIRI filters for KBO detection
PREFERRED_FILTERS = [
    'F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W', 'F1800W', 'F2100W', 'F2550W'
]

# MIRI filter wavelengths (in microns)
FILTER_WAVELENGTHS = {
    'F560W': 5.6,
    'F770W': 7.7,
    'F1000W': 10.0,
    'F1130W': 11.3,
    'F1280W': 12.8,
    'F1500W': 15.0,
    'F1800W': 18.0,
    'F2100W': 21.0,
    'F2550W': 25.5
}

def load_catalog(filename):
    """Load the full catalog JSON file"""
    print(f"Loading catalog from {filename}...")
    with open(filename, 'r') as f:
        catalog = json.load(f)
    print(f"Loaded {len(catalog)} observations")
    return catalog

def filter_by_instrument(catalog, include_nircam=False):
    """Filter for MIRI observations and optionally NIRCam observations"""
    if include_nircam:
        instrument_obs = [obs for obs in catalog 
                         if 'instrument_name' in obs and 
                         ('MIRI' in obs.get('instrument_name', '') or 
                          'NIRCAM' in obs.get('instrument_name', ''))]
        print(f"Filter 1: MIRI or NIRCam instruments - {len(instrument_obs)}/{len(catalog)} observations passed")
    else:
        instrument_obs = [obs for obs in catalog 
                         if 'instrument_name' in obs and 'MIRI' in obs.get('instrument_name', '')]
        print(f"Filter 1: MIRI instrument only - {len(instrument_obs)}/{len(catalog)} observations passed")
    return instrument_obs

def filter_by_intent(catalog, include_cal=False):
    """Filter for science observations, and optionally include calibration"""
    if include_cal:
        # Include calibration but still filter out dark and flat frames
        science_obs = [obs for obs in catalog 
                      if 'dark' not in obs.get('obs_title', '').lower() and
                         'flat' not in obs.get('obs_title', '').lower()]
        print(f"Filter 2: Including science and calibration (excluding darks/flats) - {len(science_obs)}/{len(catalog)} observations passed")
    else:
        science_obs = [obs for obs in catalog 
                      if obs.get('intentType') == 'science' or
                         (obs.get('proposal_type') != 'CAL' and 
                          'dark' not in obs.get('obs_title', '').lower() and
                          'flat' not in obs.get('obs_title', '').lower())]
        print(f"Filter 2: Science intent only - {len(science_obs)}/{len(catalog)} observations passed")
    return science_obs

def filter_by_wavelength(catalog):
    """Filter for desired wavelength range"""
    wavelength_obs = []
    for obs in catalog:
        # Check if wavelength information exists
        if 'em_min' in obs and 'em_max' in obs:
            em_min = obs.get('em_min')
            em_max = obs.get('em_max')
            
            # Handle different units or missing data
            if em_min and em_max:
                # MAST typically provides wavelengths in meters, so convert to microns
                if em_min < 1.0 and em_max < 1.0:  # Likely in meters
                    em_min_um = em_min * 1e6
                    em_max_um = em_max * 1e6
                elif em_min < 1000 and em_max < 1000:  # Likely in microns
                    em_min_um = em_min
                    em_max_um = em_max
                else:  # Likely in nanometers
                    em_min_um = em_min / 1000
                    em_max_um = em_max / 1000
                    
                # Check if wavelength range overlaps with our desired range
                if (em_min_um <= MAX_WAVELENGTH_UM and em_max_um >= MIN_WAVELENGTH_UM):
                    wavelength_obs.append(obs)
                    continue  # Skip checking filters
        
        # Also check filter names if available or if em_min/em_max not useful
        if 'filters' in obs:
            filters = str(obs.get('filters', '')).split(';')
            for filter_name in filters:
                if any(preferred in filter_name for preferred in PREFERRED_FILTERS):
                    wavelength_obs.append(obs)
                    break
        else:
            # If we have no wavelength info but made it this far, better to include than exclude
            wavelength_obs.append(obs)
    
    print(f"Filter 3: Wavelength range {MIN_WAVELENGTH_UM}-{MAX_WAVELENGTH_UM}μm - {len(wavelength_obs)}/{len(catalog)} observations passed")
    return wavelength_obs

def filter_by_exposure(catalog):
    """Filter for minimum exposure time"""
    exposure_obs = [obs for obs in catalog if obs.get('t_exptime', 0) >= MIN_EXPOSURE_TIME]
    print(f"Filter 4: Minimum exposure {MIN_EXPOSURE_TIME}s - {len(exposure_obs)}/{len(catalog)} observations passed")
    return exposure_obs

def filter_by_calib_level(catalog):
    """Filter for calibration level 2 or higher"""
    calib_obs = [obs for obs in catalog if obs.get('calib_level', 0) >= 2]
    print(f"Filter 5: Calibration level ≥2 - {len(calib_obs)}/{len(catalog)} observations passed")
    return calib_obs

def group_by_field(catalog):
    """Group observations by field (similar coordinates)"""
    fields = {}
    field_index = 0
    
    for obs in catalog:
        # Skip if no coordinates
        if 's_ra' not in obs or 's_dec' not in obs:
            continue
            
        ra = obs['s_ra']
        dec = obs['s_dec']
        
        # Check if this observation matches any existing field
        obs_coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
        field_match = False
        
        for field_id, field_data in fields.items():
            field_coord = field_data['center_coord']
            separation = obs_coord.separation(field_coord).arcsec
            
            if separation <= FIELD_MATCH_RADIUS:
                # Add to existing field
                fields[field_id]['observations'].append(obs)
                field_match = True
                break
                
        if not field_match:
            # Create new field
            fields[f"field_{field_index}"] = {
                'center_coord': obs_coord,
                'center_ra': ra,
                'center_dec': dec,
                'observations': [obs]
            }
            field_index += 1
    
    # Count fields with multiple observations
    multi_obs_fields = {field_id: data for field_id, data in fields.items() 
                        if len(data['observations']) >= MIN_SEQUENCE_IMAGES}
    
    print(f"Field grouping: Found {len(fields)} distinct fields")
    print(f"Fields with ≥{MIN_SEQUENCE_IMAGES} observations: {len(multi_obs_fields)}/{len(fields)}")
    
    return fields, multi_obs_fields

def identify_sequences(fields):
    """Identify observation sequences suitable for KBO detection"""
    sequences = []
    
    for field_id, field_data in fields.items():
        observations = field_data['observations']
        
        # Skip fields with too few observations
        if len(observations) < MIN_SEQUENCE_IMAGES:
            continue
        
        # Sort observations by time
        obs_with_time = []
        for obs in observations:
            if 't_min' in obs:
                # Convert MJD to datetime for easier handling
                t = Time(obs['t_min'], format='mjd')
                obs_with_time.append((obs, t))
                
        # Skip if we don't have time info
        if not obs_with_time:
            continue
            
        # Sort by time
        obs_with_time.sort(key=lambda x: x[1].mjd)
        
        # Find sequences with appropriate time intervals
        current_sequence = [obs_with_time[0]]
        
        for i in range(1, len(obs_with_time)):
            prev_obs, prev_time = current_sequence[-1]
            current_obs, current_time = obs_with_time[i]
            
            # Calculate time difference in hours
            time_diff_hours = (current_time.mjd - prev_time.mjd) * 24.0
            
            # Check if this observation continues the sequence
            if MIN_SEQ_INTERVAL <= time_diff_hours <= MAX_SEQ_INTERVAL:
                # Check expected KBO movement
                expected_movement = KBO_MOVEMENT_RATE * time_diff_hours
                if expected_movement <= FIELD_MATCH_RADIUS:
                    current_sequence.append((current_obs, current_time))
            else:
                # If sequence is long enough, save it and start a new one
                if len(current_sequence) >= MIN_SEQUENCE_IMAGES:
                    sequence_obs = [item[0] for item in current_sequence]
                    start_time = current_sequence[0][1].iso
                    end_time = current_sequence[-1][1].iso
                    duration_hours = (current_sequence[-1][1].mjd - current_sequence[0][1].mjd) * 24.0
                    
                    sequences.append({
                        'field_id': field_id,
                        'center_ra': field_data['center_ra'],
                        'center_dec': field_data['center_dec'],
                        'num_observations': len(sequence_obs),
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration_hours': duration_hours,
                        'observations': sequence_obs
                    })
                
                # Start a new sequence
                current_sequence = [(current_obs, current_time)]
        
        # Check if last sequence is valid
        if len(current_sequence) >= MIN_SEQUENCE_IMAGES:
            sequence_obs = [item[0] for item in current_sequence]
            start_time = current_sequence[0][1].iso
            end_time = current_sequence[-1][1].iso
            duration_hours = (current_sequence[-1][1].mjd - current_sequence[0][1].mjd) * 24.0
            
            sequences.append({
                'field_id': field_id,
                'center_ra': field_data['center_ra'],
                'center_dec': field_data['center_dec'],
                'num_observations': len(sequence_obs),
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': duration_hours,
                'observations': sequence_obs
            })
    
    # Sort sequences by number of observations (descending)
    sequences.sort(key=lambda x: x['num_observations'], reverse=True)
    
    print(f"Sequence identification: Found {len(sequences)} valid observation sequences")
    if sequences:
        print(f"  - Longest sequence: {sequences[0]['num_observations']} observations over {sequences[0]['duration_hours']:.1f} hours")
        
    return sequences

def prepare_download_catalog(sequences):
    """Prepare simplified catalog for downloader"""
    download_catalog = []
    
    for sequence in sequences:
        sequence_info = {
            'field_id': sequence['field_id'],
            'center_ra': sequence['center_ra'],
            'center_dec': sequence['center_dec'],
            'num_observations': sequence['num_observations'],
            'start_time': sequence['start_time'],
            'end_time': sequence['end_time'],
            'duration_hours': sequence['duration_hours'],
            'observations': []
        }
        
        for obs in sequence['observations']:
            # Extract only needed fields for downloader
            obs_info = {
                'obs_id': obs.get('obs_id', ''),
                'dataURL': obs.get('dataURL', ''),
                's_ra': obs.get('s_ra', 0),
                's_dec': obs.get('s_dec', 0),
                't_min': obs.get('t_min', 0),
                't_exptime': obs.get('t_exptime', 0),
                'filters': obs.get('filters', ''),
                'instrument_name': obs.get('instrument_name', ''),
                'calib_level': obs.get('calib_level', 0)
            }
            sequence_info['observations'].append(obs_info)
        
        download_catalog.append(sequence_info)
    
    return download_catalog

def generate_visualizations(fields, sequences, output_dir):
    """Generate visualizations of the observations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plotting fields on the sky
    plt.figure(figsize=(12, 8))
    for field_id, field_data in fields.items():
        obs_count = len(field_data['observations'])
        # Size points by number of observations
        size = 10 + 5 * obs_count
        # Color multi-observation fields differently
        color = 'red' if obs_count >= MIN_SEQUENCE_IMAGES else 'blue'
        plt.scatter(field_data['center_ra'], field_data['center_dec'], 
                   s=size, alpha=0.7, c=color, label=f"{field_id}: {obs_count} obs" if obs_count >= MIN_SEQUENCE_IMAGES else None)
    
    plt.xlabel('RA (degrees)')
    plt.ylabel('Dec (degrees)')
    plt.title('JWST Observation Fields')
    
    # Add legend for fields with multiple observations
    if any(len(data['observations']) >= MIN_SEQUENCE_IMAGES for data in fields.values()):
        plt.legend(title="Fields with sequences", loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'observation_fields.png'), dpi=150)
    
    if sequences:
        # Plotting sequence durations
        plt.figure(figsize=(10, 6))
        seq_durations = [seq['duration_hours'] for seq in sequences]
        seq_obs_counts = [seq['num_observations'] for seq in sequences]
        
        plt.bar(range(len(sequences)), seq_durations, alpha=0.7)
        plt.xlabel('Sequence')
        plt.ylabel('Duration (hours)')
        plt.title('Observation Sequence Durations')
        plt.xticks(range(len(sequences)), [f"{i+1}" for i in range(len(sequences))])
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'sequence_durations.png'), dpi=150)
        
        # Plotting histogram of time intervals
        plt.figure(figsize=(10, 6))
        intervals = []
        
        for seq in sequences:
            obs = seq['observations']
            if 't_min' in obs[0]:
                for i in range(1, len(obs)):
                    if 't_min' in obs[i]:
                        interval = (obs[i]['t_min'] - obs[i-1]['t_min']) * 24.0  # Convert from days to hours
                        intervals.append(interval)
        
        if intervals:
            plt.hist(intervals, bins=20, alpha=0.7)
            plt.xlabel('Time Interval (hours)')
            plt.ylabel('Count')
            plt.title('Distribution of Time Intervals Between Observations')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'time_intervals.png'), dpi=150)
    
    print(f"Visualizations saved to {output_dir}")

def main():
    import argparse
    
    # Global declarations must come before usage
    global MIN_EXPOSURE_TIME, MIN_WAVELENGTH_UM, MAX_WAVELENGTH_UM
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Filter JWST observations for KBO hunting')
    parser.add_argument('input_file', help='Path to input JSON catalog file')
    parser.add_argument('--output-dir', '-o', default="../data", 
                       help='Directory to save filtered catalog and visualizations (default: ../data)')
    parser.add_argument('--min-exposure', '-e', type=float, default=MIN_EXPOSURE_TIME,
                       help=f'Minimum exposure time in seconds (default: {MIN_EXPOSURE_TIME})')
    parser.add_argument('--min-wavelength', '-wl', type=float, default=MIN_WAVELENGTH_UM,
                       help=f'Minimum wavelength in microns (default: {MIN_WAVELENGTH_UM})')
    parser.add_argument('--max-wavelength', '-wh', type=float, default=MAX_WAVELENGTH_UM,
                       help=f'Maximum wavelength in microns (default: {MAX_WAVELENGTH_UM})')
    parser.add_argument('--include-nircam', action='store_true',
                       help='Include NIRCam observations in addition to MIRI')
    parser.add_argument('--include-cal', action='store_true',
                       help='Include calibration observations (excluding darks/flats)')
    
    args = parser.parse_args()
    
    # Update global parameters if specified on command line
    MIN_EXPOSURE_TIME = args.min_exposure
    MIN_WAVELENGTH_UM = args.min_wavelength
    MAX_WAVELENGTH_UM = args.max_wavelength
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load catalog
    print(f"Loading catalog from {args.input_file}...")
    catalog = load_catalog(args.input_file)
    
    # Apply filters
    print("\nApplying filters...")
    filtered_catalog = catalog
    filtered_catalog = filter_by_instrument(filtered_catalog, include_nircam=args.include_nircam)
    filtered_catalog = filter_by_intent(filtered_catalog, include_cal=args.include_cal)
    filtered_catalog = filter_by_wavelength(filtered_catalog)
    filtered_catalog = filter_by_exposure(filtered_catalog)
    filtered_catalog = filter_by_calib_level(filtered_catalog)
    
    print(f"\nAfter all filters: {len(filtered_catalog)}/{len(catalog)} observations passed ({len(filtered_catalog)/len(catalog)*100:.1f}%)")
    
    # Group by field and identify sequences
    print("\nGrouping observations by field...")
    fields, multi_obs_fields = group_by_field(filtered_catalog)
    
    print("\nIdentifying observation sequences...")
    sequences = identify_sequences(multi_obs_fields)
    
    # Prepare download catalog
    download_catalog = prepare_download_catalog(sequences)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"kbo_download_catalog_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(download_catalog, f, indent=2)
    
    print(f"\nSaved download catalog to {output_file}")
    print(f"Catalog contains {len(download_catalog)} sequences with a total of {sum(seq['num_observations'] for seq in download_catalog)} observations")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(fields, sequences, os.path.join(args.output_dir, "visualization"))

if __name__ == "__main__":
    main()