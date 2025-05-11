#!/usr/bin/env python3
"""
kbo_detector.py - Main script for detecting KBOs in preprocessed JWST data

This script coordinates the KBO detection process by:
1. Loading preprocessed FITS files
2. Calculating expected KBO motion vectors
3. Applying shift-and-stack for each motion vector
4. Detecting and scoring candidate KBO sources
5. Saving and visualizing the results

Can process either a single field or iterate through all field directories.
"""

import os
import sys
import json
import argparse
import numpy as np
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import simple_norm

# Create detection directory if it doesn't exist
if not os.path.exists('./detection'):
    os.makedirs('./detection')

# Import detection modules
try:
    from detection.motion_calculator import calculate_kbo_motion_range, generate_motion_vectors, filter_motion_vectors
    from detection.shift_stack import apply_shift, stack_images, process_motion_vector
    from detection.candidate_filter import filter_candidates, score_candidate, calculate_kbo_properties
    from detection.visualization import visualize_candidates, create_diagnostic_plots
except ImportError as e:
    print(f"Error importing detection modules: {e}")
    print("This could be due to missing __init__.py or module files.")
    print("Make sure all required modules are in the detection directory.")
    sys.exit(1)

# Default plate scale for JWST MIRI (arcsec/pixel)
DEFAULT_PLATE_SCALE = 0.11

def load_preprocessed_data(preprocessed_dir, verbose=True):
    """
    Load preprocessed FITS files
    
    Parameters:
    -----------
    preprocessed_dir : str
        Directory containing preprocessed FITS files
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    tuple
        (images, headers, timestamps) lists
    """
    # Normalize path for cross-platform compatibility
    preprocessed_dir = os.path.normpath(preprocessed_dir)
    
    # Look for preprocessed FITS files
    fits_files = glob(os.path.join(preprocessed_dir, "*_preprocessed.fits"))
    
    if not fits_files:
        print(f"No preprocessed FITS files found in {preprocessed_dir}")
        if verbose:
            # Add debugging information
            print(f"Absolute path: {os.path.abspath(preprocessed_dir)}")
            print(f"Directory exists: {os.path.isdir(preprocessed_dir)}")
            try:
                if os.path.isdir(preprocessed_dir):
                    print(f"Contents: {os.listdir(preprocessed_dir)}")
                else:
                    parent_dir = os.path.dirname(preprocessed_dir)
                    if os.path.isdir(parent_dir):
                        print(f"Parent directory exists. Contents: {os.listdir(parent_dir)}")
            except Exception as e:
                print(f"Error listing directory: {e}")
        return None, None, None
    
    if verbose:
        print(f"Found {len(fits_files)} preprocessed FITS files")
    
    # Load FITS files
    images = []
    headers = []
    timestamps = []
    
    for i, fits_file in enumerate(fits_files):
        try:
            if verbose:
                print(f"Loading {i+1}/{len(fits_files)}: {os.path.basename(fits_file)}")
            
            with fits.open(fits_file) as hdul:
                # Assuming the data is in the primary HDU for preprocessed files
                data = hdul[0].data
                header = hdul[0].header
                
                # Check for NaN values
                nan_fraction = np.sum(np.isnan(data)) / data.size
                if nan_fraction > 0.9:  # Skip if more than 90% NaN
                    if verbose:
                        print(f"  Warning: File contains {nan_fraction*100:.1f}% NaN values. Skipping.")
                    continue
                
                # Extract timestamp from header
                if 'MJD-OBS' in header:
                    timestamp = header['MJD-OBS']
                elif 'MJD' in header:
                    timestamp = header['MJD']
                elif 'DATE-OBS' in header:
                    time_obj = Time(header['DATE-OBS'], format='isot', scale='utc')
                    timestamp = time_obj.mjd
                else:
                    # If no timestamp found, use relative time based on order
                    print(f"  Warning: No timestamp found in header. Using arbitrary value.")
                    timestamp = float(i)
                
                images.append(data)
                headers.append(header)
                timestamps.append(timestamp)
                
                if verbose:
                    print(f"  Shape: {data.shape}")
                    print(f"  Timestamp: {timestamp}")
                    print(f"  NaN values: {nan_fraction*100:.1f}%")
        
        except Exception as e:
            print(f"Error loading {fits_file}: {e}")
    
    if not images:
        print("No valid preprocessed images found!")
        return None, None, None
    
    # Sort images by timestamp
    sorted_indices = np.argsort(timestamps)
    images = [images[i] for i in sorted_indices]
    headers = [headers[i] for i in sorted_indices]
    timestamps = [timestamps[i] for i in sorted_indices]
    
    if verbose:
        print(f"Successfully loaded {len(images)} preprocessed images")
        
        # Show time intervals
        if len(timestamps) > 1:
            print("\nTime intervals from first image:")
            for i in range(1, len(timestamps)):
                hours = (timestamps[i] - timestamps[0]) * 24.0
                print(f"  Image {i+1}: {hours:.2f} hours ({hours/24:.2f} days)")
    
    return images, headers, timestamps

def get_plate_scale(headers, default=DEFAULT_PLATE_SCALE):
    """
    Get plate scale from FITS headers
    
    Parameters:
    -----------
    headers : list
        List of FITS headers
    default : float
        Default plate scale to use if not found in headers
    
    Returns:
    --------
    float
        Plate scale in arcsec/pixel
    """
    for header in headers:
        # Try common keywords for plate scale
        if 'PIXSCALE' in header:
            return header['PIXSCALE']
        elif 'CDELT1' in header:
            # CDELT1 is typically in degrees/pixel
            return abs(header['CDELT1']) * 3600.0
        elif 'CD1_1' in header and 'CD2_2' in header:
            # CD matrix - take average of absolute values
            return (abs(header['CD1_1']) + abs(header['CD2_2'])) / 2.0 * 3600.0
    
    # If no plate scale found, use default
    print(f"Warning: Could not find plate scale in headers. Using default: {default} arcsec/pixel")
    return default

def detect_moving_objects(images, timestamps, output_dir=None, visualization_dir=None, verbose=True):
    """
    Detect moving objects in a sequence of preprocessed FITS files
    
    Parameters:
    -----------
    images : list
        List of image data arrays
    timestamps : list
        List of MJD timestamps
    output_dir : str
        Output directory for results
    visualization_dir : str
        Output directory for visualizations
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    dict
        Results including candidate objects
    """
    # Normalize output paths
    if output_dir:
        output_dir = os.path.normpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    if visualization_dir:
        visualization_dir = os.path.normpath(visualization_dir)
        os.makedirs(visualization_dir, exist_ok=True)
    
    if verbose:
        print(f"\n=== Detecting KBOs in {len(images)} Images ===\n")
    
    if len(images) < 2:
        print("Need at least 2 images for motion detection!")
        return None
    
    # Calculate time intervals in hours
    time_intervals = [(t - timestamps[0]) * 24.0 for t in timestamps[1:]]
    
    if verbose:
        print("Time intervals from first image (hours):")
        for i, interval in enumerate(time_intervals):
            print(f"  Image {i+2}: {interval:.2f} hours")
    
    # Get plate scale
    plate_scale = DEFAULT_PLATE_SCALE  # Default value - would ideally come from FITS headers
    
    # Calculate expected KBO motion range
    total_time_hours = time_intervals[-1] if time_intervals else 1.0
    motion_range = calculate_kbo_motion_range(total_time_hours, plate_scale, verbose)
    
    # Generate motion vectors to test
    raw_motion_vectors = generate_motion_vectors(
        motion_range['min_motion_pixels'] / len(images),
        motion_range['max_motion_pixels'] / len(images),
        num_steps=8,
        num_angles=12,
        verbose=verbose
    )
    
    # Filter motion vectors to reasonable values
    motion_vectors = filter_motion_vectors(
        raw_motion_vectors,
        plate_scale,
        total_time_hours,
        max_vectors=100
    )
    
    if verbose:
        print(f"After filtering: Testing {len(motion_vectors)} motion vectors")
    
    # Initial shifts for stack alignment (all zeros for now)
    base_shifts = [(0, 0) for _ in range(len(images))]
    
    if verbose:
        print("\nRunning shift-and-stack detection...")
        print(f"  Testing {len(motion_vectors)} motion vectors")
    
    # Candidate objects from all motion vectors
    all_candidates = []
    
    # Process each motion vector
    for i, motion_vector in enumerate(motion_vectors):
        if verbose and (i % 10 == 0 or i == len(motion_vectors) - 1):
            print(f"  Processing motion vector {i+1}/{len(motion_vectors)}: dx={motion_vector[0]:.2f}, dy={motion_vector[1]:.2f}")
        
        # Process this motion vector
        stacked, sources, shifts = process_motion_vector(images, base_shifts, motion_vector)
        
        if sources is not None and len(sources) > 0:
            # For each detected source, create a candidate
            for j, source in enumerate(sources):
                candidate = {
                    'motion_vector_idx': i,
                    'source_idx': j,
                    'motion_vector': motion_vector,
                    'xcentroid': source['xcentroid'],
                    'ycentroid': source['ycentroid'],
                    'flux': source['flux'],
                    'peak': source['peak'],
                    'shifts': shifts
                }
                
                # Score the candidate
                candidate['score'] = score_candidate(candidate, images, 
                                                   [(s[0] - shifts[0][0], s[1] - shifts[0][1]) for s in shifts])
                
                all_candidates.append(candidate)
    
    if verbose:
        print(f"\nFound {len(all_candidates)} initial candidates")
    
    # Filter and sort candidates - use physics-based filtering
    if all_candidates:
        # Filter candidates based on physical constraints
        filtered_candidates = filter_candidates(
            all_candidates, 
            total_time_hours,
            plate_scale, 
            verbose
        )
        
        # Add physical properties to remaining candidates
        for candidate in filtered_candidates:
            properties = calculate_kbo_properties(
                candidate,
                plate_scale,
                total_time_hours
            )
            candidate.update(properties)
        
        if verbose:
            print(f"After filtering: {len(filtered_candidates)} candidates remain")
            if filtered_candidates:
                print("\nTop candidates:")
                for i, candidate in enumerate(filtered_candidates[:5]):
                    print(f"  {i+1}. Score: {candidate['score']:.2f}, Position: ({candidate['xcentroid']:.1f}, {candidate['ycentroid']:.1f})")
                    print(f"     Motion: {candidate['motion_arcsec_per_hour']:.2f} arcsec/hour, Est. dist: {candidate['approx_distance_au']:.1f} AU")
    else:
        filtered_candidates = []
        if verbose:
            print("No candidates found")
    
    # Prepare results
    results = {
        'timestamp': datetime.now().isoformat(),
        'num_images': len(images),
        'time_span_hours': total_time_hours,
        'plate_scale': plate_scale,
        'motion_vectors_tested': len(motion_vectors),
        'initial_candidates': len(all_candidates),
        'filtered_candidates': len(filtered_candidates),
        'candidates': filtered_candidates,
        'motion_range': {
            'min_motion_arcsec': motion_range['min_motion_arcsec'],
            'max_motion_arcsec': motion_range['max_motion_arcsec'],
            'min_motion_pixels': motion_range['min_motion_pixels'],
            'max_motion_pixels': motion_range['max_motion_pixels']
        }
    }
    
    # Save results if requested
    if output_dir and filtered_candidates:
        results_file = os.path.join(output_dir, "kbo_candidates.json")
        
        # Convert numpy values to Python types for JSON serialization
        serializable_results = {
            'timestamp': results['timestamp'],
            'num_images': results['num_images'],
            'time_span_hours': float(results['time_span_hours']),
            'plate_scale': float(results['plate_scale']),
            'motion_vectors_tested': results['motion_vectors_tested'],
            'initial_candidates': results['initial_candidates'],
            'filtered_candidates': results['filtered_candidates'],
            'motion_range': {
                'min_motion_arcsec': float(results['motion_range']['min_motion_arcsec']),
                'max_motion_arcsec': float(results['motion_range']['max_motion_arcsec']),
                'min_motion_pixels': float(results['motion_range']['min_motion_pixels']),
                'max_motion_pixels': float(results['motion_range']['max_motion_pixels'])
            },
            'candidates': []
        }
        
        for candidate in filtered_candidates:
            serializable_candidate = {
                'motion_vector_idx': candidate['motion_vector_idx'],
                'source_idx': candidate['source_idx'],
                'motion_vector': [float(candidate['motion_vector'][0]), float(candidate['motion_vector'][1])],
                'xcentroid': float(candidate['xcentroid']),
                'ycentroid': float(candidate['ycentroid']),
                'flux': float(candidate['flux']),
                'peak': float(candidate['peak']),
                'score': float(candidate['score']),
                'motion_arcsec_per_hour': float(candidate['motion_arcsec_per_hour']),
                'approx_distance_au': float(candidate['approx_distance_au']),
                'shifts': [[float(s[0]), float(s[1])] for s in candidate['shifts']]
            }
            serializable_results['candidates'].append(serializable_candidate)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if verbose:
            print(f"\nSaved results to {results_file}")
    
    # Generate visualizations if requested
    if visualization_dir and filtered_candidates:
        visualize_candidates(
            images, 
            filtered_candidates, 
            visualization_dir, 
            time_span_hours=total_time_hours,
            plate_scale=plate_scale,
            verbose=verbose
        )
        
        # Create diagnostic plots
        create_diagnostic_plots(
            motion_range,
            filtered_candidates,
            visualization_dir
        )
    
    return results

def find_field_directories(preprocessed_dir):
    """
    Find all field directories in the preprocessed directory
    
    Parameters:
    -----------
    preprocessed_dir : str
        Base directory containing preprocessed FITS files
        
    Returns:
    --------
    list
        List of field directory paths and IDs
    """
    # Check if the directory exists
    if not os.path.isdir(preprocessed_dir):
        print(f"Error: Preprocessed directory {preprocessed_dir} not found")
        return []
    
    # Look for subdirectories containing preprocessed FITS files
    field_dirs = []
    
    # First check if the main directory contains FITS files directly
    main_fits_files = glob(os.path.join(preprocessed_dir, "*_preprocessed.fits"))
    if main_fits_files:
        field_dirs.append(('main', preprocessed_dir))
    
    # Then check all subdirectories
    for item in os.listdir(preprocessed_dir):
        subdir = os.path.join(preprocessed_dir, item)
        if os.path.isdir(subdir):
            # Check for preprocessed files in this subdirectory
            fits_files = glob(os.path.join(subdir, "*_preprocessed.fits"))
            if fits_files:
                field_dirs.append((item, subdir))
            else:
                # Check if there are deeper subdirectories with FITS files (sometimes the structure is deeper)
                for subitem in os.listdir(subdir) if os.path.isdir(subdir) else []:
                    subsubdir = os.path.join(subdir, subitem)
                    if os.path.isdir(subsubdir):
                        sub_fits_files = glob(os.path.join(subsubdir, "*_preprocessed.fits"))
                        if sub_fits_files:
                            # Use combined path as field ID
                            field_id = f"{item}/{subitem}"
                            field_dirs.append((field_id, subsubdir))
    
    return field_dirs

def create_summary_report(all_results, output_dir):
    """
    Create a summary report of all processed fields
    
    Parameters:
    -----------
    all_results : dict
        Dictionary mapping field IDs to results
    output_dir : str
        Output directory for the summary report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Count total candidates
    total_candidates = 0
    for field_id, result in all_results.items():
        if result:
            total_candidates += result.get('filtered_candidates', 0)
    
    # Create summary data
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_fields_processed': len(all_results),
        'fields_with_candidates': sum(1 for r in all_results.values() if r and r.get('filtered_candidates', 0) > 0),
        'total_candidates': total_candidates,
        'field_results': {}
    }
    
    for field_id, result in all_results.items():
        if result:
            field_name = os.path.basename(field_id)
            summary['field_results'][field_name] = {
                'num_images': result.get('num_images', 0),
                'time_span_hours': float(result.get('time_span_hours', 0)),
                'initial_candidates': result.get('initial_candidates', 0),
                'filtered_candidates': result.get('filtered_candidates', 0)
            }
            
            # Add candidate details
            if 'candidates' in result and result['candidates']:
                candidates_info = []
                for candidate in result['candidates']:
                    candidates_info.append({
                        'score': float(candidate.get('score', 0)),
                        'motion_arcsec_per_hour': float(candidate.get('motion_arcsec_per_hour', 0)),
                        'approx_distance_au': float(candidate.get('approx_distance_au', 0))
                    })
                summary['field_results'][field_name]['candidates'] = candidates_info
    
    # Save summary to file
    summary_file = os.path.join(output_dir, "detection_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create a simple text report
    report_file = os.path.join(output_dir, "detection_report.txt")
    with open(report_file, 'w') as f:
        f.write("KBO DETECTION SUMMARY\n")
        f.write("====================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Fields processed: {len(all_results)}\n")
        f.write(f"Fields with candidates: {summary['fields_with_candidates']}\n")
        f.write(f"Total candidates: {total_candidates}\n\n")
        
        f.write("RESULTS BY FIELD\n")
        f.write("===============\n\n")
        
        # Sort fields by number of candidates (descending)
        sorted_fields = sorted(
            [(field_id, result) for field_id, result in all_results.items() if result],
            key=lambda x: x[1].get('filtered_candidates', 0),
            reverse=True
        )
        
        for field_id, result in sorted_fields:
            f.write(f"Field: {field_id}\n")
            f.write(f"  Images: {result.get('num_images', 0)}\n")
            f.write(f"  Time span: {result.get('time_span_hours', 0):.2f} hours\n")
            f.write(f"  Initial candidates: {result.get('initial_candidates', 0)}\n")
            f.write(f"  Filtered candidates: {result.get('filtered_candidates', 0)}\n")
            
            if result.get('candidates'):
                f.write("  Top candidates:\n")
                for i, candidate in enumerate(result['candidates'][:5]):  # Show top 5
                    f.write(f"    {i+1}. Score: {candidate['score']:.2f}, " 
                           f"Motion: {candidate.get('motion_arcsec_per_hour', 0):.2f} arcsec/hour, "
                           f"Distance: {candidate.get('approx_distance_au', 0):.1f} AU\n")
            
            f.write("\n")
    
    print(f"\nSummary report saved to:")
    print(f"  - {summary_file}")
    print(f"  - {report_file}")
    
    return summary_file, report_file

def main():
    parser = argparse.ArgumentParser(description="Detect KBOs in preprocessed JWST FITS files")
    parser.add_argument('--preprocessed-dir', default="./data/preprocessed", 
                       help="Directory containing preprocessed FITS files (default: ./data/preprocessed)")
    parser.add_argument('--field-id', help="Field ID to process (subfolder of preprocessed-dir)")
    parser.add_argument('--all-fields', action='store_true', 
                       help="Process all field directories in the preprocessed directory")
    parser.add_argument('--output-dir', default="./data/detections", 
                       help="Output directory for detection results (default: ./data/detections)")
    parser.add_argument('--visualization-dir', default="./data/visualization/detections", 
                       help="Output directory for visualizations (default: ./data/visualization/detections)")
    parser.add_argument('--verbose', '-v', action='store_true', help="Print verbose output")
    parser.add_argument('--list-fields', action='store_true', 
                       help="List all available fields and exit")
    parser.add_argument('--min-score', type=float, default=0.7,
                       help="Minimum score threshold for candidates (default: 0.7)")
    
    args = parser.parse_args()
    
    # Update global constants based on command line arguments
    if args.min_score != 0.7:
        from detection.candidate_filter import KBO_CONSTRAINTS
        KBO_CONSTRAINTS['min_score'] = args.min_score
    
    preprocessed_dir = os.path.normpath(args.preprocessed_dir)
    output_dir = os.path.normpath(args.output_dir)
    vis_dir = os.path.normpath(args.visualization_dir)
    
    # Find all available fields first
    available_fields = find_field_directories(preprocessed_dir)
    
    # Just list fields if requested
    if args.list_fields:
        if not available_fields:
            print(f"No fields with preprocessed FITS files found in {preprocessed_dir}")
        else:
            print(f"Available fields in {preprocessed_dir}:")
            for field_id, field_path in available_fields:
                num_files = len(glob(os.path.join(field_path, "*_preprocessed.fits")))
                print(f"  {field_id}: {num_files} preprocessed files")
        return
    
    # Determine fields to process
    fields_to_process = []
    
    if args.field_id:
        # Process a single specified field
        # First check if it's in the available fields
        matching_fields = [f for f in available_fields if f[0] == args.field_id]
        if matching_fields:
            fields_to_process.append(matching_fields[0])
        else:
            # Try looking in subdirectory
            field_path = os.path.join(preprocessed_dir, args.field_id)
            if os.path.isdir(field_path) and glob(os.path.join(field_path, "*_preprocessed.fits")):
                fields_to_process.append((args.field_id, field_path))
            else:
                print(f"Field '{args.field_id}' not found or contains no preprocessed FITS files")
                print("Available fields:")
                for field_id, _ in available_fields:
                    print(f"  {field_id}")
                return
    elif args.all_fields:
        # Process all available fields
        fields_to_process = available_fields
    else:
        # Default: if there are any fields, process them all
        if available_fields:
            fields_to_process = available_fields
        else:
            print(f"No fields with preprocessed FITS files found in {preprocessed_dir}")
            print("Please use --list-fields to see available fields")
            return
    
    if not fields_to_process:
        print(f"No valid fields found in {preprocessed_dir}")
        return
    
    print(f"Found {len(fields_to_process)} fields to process")
    
    # Process each field
    all_results = {}
    
    for field_id, field_path in fields_to_process:
        print(f"\n{'='*80}")
        print(f"Processing field: {field_id}")
        print(f"{'='*80}\n")
        
        # Set up output directories for this field
        field_output_dir = os.path.join(output_dir, field_id)
        field_vis_dir = os.path.join(vis_dir, field_id)
        
        # Load preprocessed images
        images, headers, timestamps = load_preprocessed_data(field_path, args.verbose)
        
        if not images:
            print(f"No valid preprocessed images found in field {field_id}. Skipping.")
            all_results[field_id] = None
            continue
        
        # Run detection
        results = detect_moving_objects(
            images, 
            timestamps, 
            field_output_dir, 
            field_vis_dir, 
            args.verbose
        )
        
        all_results[field_id] = results
        
        if results and results.get('filtered_candidates', 0) > 0:
            print(f"\nDetection complete for field {field_id}!")
            print(f"Found {results['filtered_candidates']} potential KBO candidates.")
            print(f"Results saved to {field_output_dir}")
            print(f"Visualizations saved to {field_vis_dir}")
        else:
            print(f"\nNo KBO candidates detected in field {field_id}")
    
    # Create summary report
    if len(fields_to_process) > 1:
        create_summary_report(all_results, output_dir)
    
    # Print final summary
    total_candidates = sum(r.get('filtered_candidates', 0) for r in all_results.values() if r)
    print(f"\nAll processing complete! Processed {len(fields_to_process)} fields.")
    print(f"Total KBO candidates found: {total_candidates}")

if __name__ == "__main__":
    main()