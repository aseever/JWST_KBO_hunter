#!/usr/bin/env python3
"""
kbo_detector.py - Main script for detecting KBOs in preprocessed JWST data

This script coordinates the KBO detection process by:
1. Loading preprocessed FITS files
2. Calculating expected KBO motion vectors
3. Applying shift-and-stack for each motion vector
4. Detecting and scoring candidate KBO sources
5. Saving and visualizing the results
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

# Import detection modules
from detection.motion_grid import calculate_kbo_motion_range, generate_motion_vectors
from detection.shift_stack import apply_shift, stack_images, process_motion_vector
from detection.source_finder import detect_sources, filter_sources, score_candidate

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
    # Look for preprocessed FITS files
    fits_files = glob(os.path.join(preprocessed_dir, "*_preprocessed.fits"))
    
    if not fits_files:
        print(f"No preprocessed FITS files found in {preprocessed_dir}")
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
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if visualization_dir:
        os.makedirs(visualization_dir, exist_ok=True)
    
    if verbose:
        print(f"\n=== Detecting Moving Objects in {len(images)} Images ===\n")
    
    if len(images) < 2:
        print("Need at least 2 images for motion detection!")
        return None
    
    # Calculate time intervals in hours
    time_intervals = [(t - timestamps[0]) * 24.0 for t in timestamps[1:]]
    
    if verbose:
        print("Time intervals from first image (hours):")
        for i, interval in enumerate(time_intervals):
            print(f"  Image {i+2}: {interval:.2f} hours")
    
    # Calculate expected KBO motion range
    total_time_hours = time_intervals[-1] if time_intervals else 1.0
    motion_range = calculate_kbo_motion_range(total_time_hours, verbose)
    
    # Generate motion vectors to test
    motion_vectors = generate_motion_vectors(
        motion_range['min_motion_pixels'] / len(images),
        motion_range['max_motion_pixels'] / len(images),
        num_steps=8,
        num_angles=12,
        verbose=verbose
    )
    
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
    
    # Filter and sort candidates
    if all_candidates:
        # Remove candidates with low scores
        filtered_candidates = [c for c in all_candidates if c['score'] > 0.5]
        
        # Sort by score (descending)
        filtered_candidates.sort(key=lambda c: c['score'], reverse=True)
        
        if verbose:
            print(f"After filtering: {len(filtered_candidates)} candidates")
            if filtered_candidates:
                print("\nTop candidates:")
                for i, candidate in enumerate(filtered_candidates[:5]):
                    print(f"  {i+1}. Score: {candidate['score']:.2f}, Position: ({candidate['xcentroid']:.1f}, {candidate['ycentroid']:.1f})")
                    print(f"     Motion: dx={candidate['motion_vector'][0]:.3f}, dy={candidate['motion_vector'][1]:.3f} pixels/image")
    else:
        filtered_candidates = []
        if verbose:
            print("No candidates found")
    
    # Prepare results
    results = {
        'timestamp': datetime.now().isoformat(),
        'num_images': len(images),
        'time_span_hours': total_time_hours,
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
        results_file = os.path.join(output_dir, "moving_object_candidates.json")
        
        # Convert numpy values to Python types for JSON serialization
        serializable_results = {
            'timestamp': results['timestamp'],
            'num_images': results['num_images'],
            'time_span_hours': float(results['time_span_hours']),
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
                'shifts': [[float(s[0]), float(s[1])] for s in candidate['shifts']]
            }
            serializable_results['candidates'].append(serializable_candidate)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        if verbose:
            print(f"\nSaved results to {results_file}")
    
    # Generate visualizations if requested
    if visualization_dir and filtered_candidates:
        visualize_candidates(images, filtered_candidates, visualization_dir, verbose)
    
    return results

def visualize_candidates(images, candidates, output_dir, verbose=True):
    """
    Generate visualizations for candidate moving objects
    
    Parameters:
    -----------
    images : list
        List of image data arrays
    candidates : list
        List of candidate objects
    output_dir : str
        Output directory for visualizations
    verbose : bool
        Whether to print verbose information
    """
    if verbose:
        print("\nGenerating visualizations for top candidates...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Display settings for images
    image_size = 40  # Size of cutout in pixels
    
    # Visualize top candidates (up to 10)
    for i, candidate in enumerate(candidates[:10]):
        if verbose:
            print(f"  Generating visualization for candidate {i+1}")
        
        # Extract candidate info
        x, y = candidate['xcentroid'], candidate['ycentroid']
        motion_vector = candidate['motion_vector']
        shifts = candidate['shifts']
        
        # Create figure for before/after
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left panel: Original first image with predicted path
        half_size = image_size // 2
        
        # Extract cutout from first image
        x_min = max(0, int(x - half_size))
        x_max = min(images[0].shape[1], int(x + half_size))
        y_min = max(0, int(y - half_size))
        y_max = min(images[0].shape[0], int(y + half_size))
        
        cutout_orig = images[0][y_min:y_max, x_min:x_max]
        
        # Display cutout
        norm = simple_norm(cutout_orig, 'sqrt', percent=99)
        axes[0].imshow(cutout_orig, origin='lower', cmap='viridis', norm=norm)
        
        # Mark the predicted path
        path_x = []
        path_y = []
        for j in range(len(images)):
            # Reverse the shift to get the position in the first image
            pos_x = x - (j * motion_vector[0])
            pos_y = y - (j * motion_vector[1])
            
            # Adjust for cutout coordinates
            cutout_x = pos_x - x_min
            cutout_y = pos_y - y_min
            
            path_x.append(cutout_x)
            path_y.append(cutout_y)
        
        # Plot path on first image
        axes[0].plot(path_x, path_y, 'r-', alpha=0.8)
        for j, (px, py) in enumerate(zip(path_x, path_y)):
            # Circle for each expected position
            axes[0].plot(px, py, 'ro', alpha=0.6, markersize=8)
            # Add small text label
            axes[0].text(px + 2, py + 2, str(j+1), color='white', fontsize=8)
        
        axes[0].set_title("First image with predicted path")
        
        # Right panel: Stacked image
        # Stack the images according to the shifts
        stacked = stack_images(images, shifts)
        
        # Extract cutout from stacked image
        cutout_stack = stacked[y_min:y_max, x_min:x_max]
        
        # Display cutout
        norm = simple_norm(cutout_stack, 'sqrt', percent=99)
        axes[1].imshow(cutout_stack, origin='lower', cmap='viridis', norm=norm)
        
        # Mark the candidate position
        stack_x = x - x_min
        stack_y = y - y_min
        axes[1].plot(stack_x, stack_y, 'ro', alpha=0.8, markersize=10)
        
        axes[1].set_title("Stacked image with candidate position")
        
        # Add motion vector info
        dx, dy = motion_vector
        speed = math.sqrt(dx*dx + dy*dy) * len(images)
        angle = math.degrees(math.atan2(dy, dx))
        
        fig.suptitle(f"Candidate {i+1}: Score {candidate['score']:.2f}, Motion {speed:.1f} pixels at {angle:.1f}°", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f"candidate_{i+1:02d}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"    Saved to {output_file}")
    
    # Create summary figure with all top candidates
    if len(candidates) > 0:
        if verbose:
            print("  Generating summary figure...")
        
        num_candidates = min(10, len(candidates))
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, candidate in enumerate(candidates[:num_candidates]):
            if i < len(axes):
                # Extract candidate info
                x, y = candidate['xcentroid'], candidate['ycentroid']
                shifts = candidate['shifts']
                
                # Stack images with this candidate's shifts
                stacked = stack_images(images, shifts)
                
                # Extract cutout
                half_size = image_size // 2
                x_min = max(0, int(x - half_size))
                x_max = min(stacked.shape[1], int(x + half_size))
                y_min = max(0, int(y - half_size))
                y_max = min(stacked.shape[0], int(y + half_size))
                
                cutout = stacked[y_min:y_max, x_min:x_max]
                
                # Display cutout
                norm = simple_norm(cutout, 'sqrt', percent=99)
                axes[i].imshow(cutout, origin='lower', cmap='viridis', norm=norm)
                
                # Mark the candidate position
                cutout_x = x - x_min
                cutout_y = y - y_min
                axes[i].plot(cutout_x, cutout_y, 'ro', alpha=0.8, markersize=6)
                
                # Add score
                axes[i].set_title(f"Candidate {i+1}: Score {candidate['score']:.2f}")
                
                # Remove axis ticks
                axes[i].set_xticks([])
                axes[i].set_yticks([])
        
        # Hide unused subplots
        for i in range(num_candidates, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("Top Moving Object Candidates", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, "candidate_summary.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"    Saved summary to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Detect KBOs in preprocessed JWST FITS files")
    parser.add_argument('--preprocessed-dir', required=True, help="Directory containing preprocessed FITS files")
    parser.add_argument('--field-id', help="Field ID to process (subfolder of preprocessed-dir)")
    parser.add_argument('--output-dir', default="./data/detections", help="Output directory for detection results")
    parser.add_argument('--visualization-dir', default="./data/visualization/detections", 
                       help="Output directory for visualizations")
    parser.add_argument('--verbose', '-v', action='store_true', help="Print verbose output")
    
    args = parser.parse_args()
    
    # If field_id is provided, look in that subfolder
    if args.field_id:
        preprocessed_dir = os.path.join(args.preprocessed_dir, args.field_id)
        output_subdir = os.path.join(args.output_dir, args.field_id)
        vis_subdir = os.path.join(args.visualization_dir, args.field_id)
    else:
        preprocessed_dir = args.preprocessed_dir
        output_subdir = args.output_dir
        vis_subdir = args.visualization_dir
    
    # Load preprocessed images
    images, headers, timestamps = load_preprocessed_data(preprocessed_dir, args.verbose)
    
    if not images:
        print("No valid preprocessed images found. Aborting detection.")
        return
    
    # Run detection
    results = detect_moving_objects(
        images, 
        timestamps, 
        output_subdir, 
        vis_subdir, 
        args.verbose
    )
    
    if results and results['filtered_candidates'] > 0:
        print(f"\nDetection complete! Found {results['filtered_candidates']} candidate moving objects.")
        print(f"Results saved to {os.path.join(output_subdir, 'moving_object_candidates.json')}")
        print(f"Visualizations saved to {vis_subdir}")
    else:
        print("\nNo moving objects detected in this field")

if __name__ == "__main__":
    main()