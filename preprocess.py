#!/usr/bin/env python3
"""
preprocess.py - Main script for preprocessing JWST FITS files

This script uses the modules in the preprocessing package to:
1. Load and validate FITS files
2. Perform background subtraction and clean bad pixels
3. Align images to a common reference frame
4. Save preprocessed data for KBO detection

Can process either a single field or iterate through all field directories.
"""

import os
import sys
import json
import argparse
import numpy as np
import time
import threading
import traceback
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from astropy.io import fits

# Import preprocessing modules
from preprocessing.fits_loader import load_fits_file
from preprocessing.calibration import subtract_background, clean_image
from preprocessing.alignment import align_images, stack_aligned_images

# Thread-based timeout mechanism (works on all platforms including Windows)
class TimeoutError(Exception):
    """Exception raised when a function call times out."""
    pass

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=300):
    """
    Run a function with a timeout using threads (cross-platform solution)
    
    Parameters:
    -----------
    func : callable
        The function to run
    args : tuple
        Positional arguments for the function
    kwargs : dict
        Keyword arguments for the function
    timeout_seconds : int
        Timeout in seconds
        
    Returns:
    --------
    The function result
    
    Raises:
    -------
    TimeoutError if the function times out
    """
    result = [None]
    error = [None]
    finished = [False]
    
    def worker():
        try:
            result[0] = func(*args, **kwargs)
            finished[0] = True
        except Exception as e:
            error[0] = e
            finished[0] = True
    
    thread = threading.Thread(target=worker)
    thread.daemon = True  # Thread will be killed when main program exits
    
    thread.start()
    thread.join(timeout_seconds)
    
    if finished[0]:
        if error[0] is not None:
            raise error[0]
        return result[0]
    else:
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")

def save_preprocessed_data(images, output_dir):
    """Save preprocessed images for later use"""
    # Normalize path to ensure consistent separator handling
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving preprocessed data to {output_dir}")
    
    # Create a metadata file with information about the preprocessing
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'num_images': len(images),
        'images': []
    }
    
    for i, image in enumerate(images):
        # Create output filename
        basename = os.path.splitext(image['filename'])[0]
        output_file = os.path.normpath(os.path.join(output_dir, f"{basename}_preprocessed.fits"))
        
        try:
            # Create a new FITS file with the preprocessed data
            hdu = fits.PrimaryHDU(image['aligned_data'])
            
            # Copy important header information - safely
            for key in image['primary_header']:
                if key not in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND']:
                    try:
                        # Skip keys with problematic values
                        if isinstance(image['primary_header'][key], str) and '\n' in image['primary_header'][key]:
                            continue
                        hdu.header[key] = image['primary_header'][key]
                    except (ValueError, TypeError):
                        # Skip headers that can't be properly represented in FITS
                        continue
            
            # Add preprocessing information
            SENTINEL_VALUE = -999.0
            
            # Add basic preprocessing flags
            hdu.header['PREPROC'] = True
            hdu.header['ALIGNED'] = 'aligned_data' in image
            
            # Add background level if available (safely)
            if 'bg_level' in image and image['bg_level'] is not None:
                try:
                    bg_level = float(image['bg_level'])
                    if np.isnan(bg_level) or np.isinf(bg_level):
                        bg_level = SENTINEL_VALUE
                    hdu.header['BG_LEVEL'] = bg_level
                except (ValueError, TypeError):
                    hdu.header['BG_LEVEL'] = SENTINEL_VALUE
            
            # Add background noise if available (safely)
            if 'bg_noise' in image and image['bg_noise'] is not None:
                try:
                    bg_noise = float(image['bg_noise'])
                    if np.isnan(bg_noise) or np.isinf(bg_noise):
                        bg_noise = SENTINEL_VALUE
                    hdu.header['BG_NOISE'] = bg_noise
                except (ValueError, TypeError):
                    hdu.header['BG_NOISE'] = SENTINEL_VALUE
            
            # Add alignment info if available
            if 'alignment_shift' in image and image['alignment_shift'] is not None:
                try:
                    dx, dy = image['alignment_shift']
                    dx = float(dx)
                    dy = float(dy)
                    if np.isnan(dx) or np.isinf(dx):
                        dx = SENTINEL_VALUE
                    if np.isnan(dy) or np.isinf(dy):
                        dy = SENTINEL_VALUE
                    hdu.header['ALIGN_DX'] = dx
                    hdu.header['ALIGN_DY'] = dy
                except (ValueError, TypeError, IndexError):
                    hdu.header['ALIGN_DX'] = SENTINEL_VALUE
                    hdu.header['ALIGN_DY'] = SENTINEL_VALUE
            
            # Save the FITS file
            hdu.writeto(output_file, overwrite=True)
            print(f"  Saved {output_file}")
            
            # Create a quick preview image
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Handle the case where image contains only NaN values
                valid_data = image['aligned_data'][~np.isnan(image['aligned_data'])]
                if len(valid_data) > 0:
                    vmin, vmax = np.percentile(valid_data, [1, 99])
                    im = ax.imshow(image['aligned_data'], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                else:
                    # If all values are NaN, show a blank image
                    im = ax.imshow(image['aligned_data'], origin='lower', cmap='viridis')
                    ax.text(0.5, 0.5, "All NaN values", ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    
                plt.colorbar(im, ax=ax, label='Flux')
                ax.set_title(f"Preprocessed: {image['filename']}")
                preview_file = os.path.normpath(os.path.join(output_dir, f"{basename}_preview.png"))
                plt.savefig(preview_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # Add to metadata
                metadata['images'].append({
                    'index': i,
                    'filename': image['filename'],
                    'output_file': os.path.basename(output_file),
                    'preview_file': os.path.basename(preview_file),
                    'instrument': image.get('instrument', 'Unknown'),
                    'filter': image.get('filter', 'Unknown'),
                    'exptime': float(image.get('exptime', 0)) if not np.isnan(image.get('exptime', 0)) else 0,
                    'bg_level': float(image.get('bg_level', 0)) if not np.isnan(image.get('bg_level', 0)) else 0,
                    'bg_noise': float(image.get('bg_noise', 0)) if not np.isnan(image.get('bg_noise', 0)) else 0,
                    'shape': list(image['aligned_data'].shape) if 'aligned_data' in image else []
                })
                
            except Exception as e:
                print(f"  Warning: Error creating preview image: {e}")
                # Add minimal metadata
                metadata['images'].append({
                    'index': i,
                    'filename': image['filename'],
                    'output_file': os.path.basename(output_file),
                    'error': f"Preview generation failed: {str(e)}"
                })
        
        except Exception as e:
            print(f"  Error saving FITS file {output_file}: {e}")
            # Add error metadata
            metadata['images'].append({
                'index': i,
                'filename': image['filename'],
                'error': f"FITS saving failed: {str(e)}"
            })
    
    # Save metadata
    try:
        metadata_file = os.path.normpath(os.path.join(output_dir, "preprocessing_metadata.json"))
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata to {metadata_file}")
    except Exception as e:
        print(f"  Error saving metadata: {e}")
        metadata_file = None
    
    return metadata_file

def visualize_preprocessing(images, output_dir):
    """Create before/after visualizations of the preprocessing steps"""
    # Normalize path
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating preprocessing visualizations...")
    
    for i, image in enumerate(images):
        try:
            print(f"  Creating visualization for image {i}: {image['filename']}")
            
            # Check if required keys exist
            if not all(k in image for k in ['data', 'cleaned_data', 'aligned_data']):
                print(f"    Missing required data for image {i}, skipping visualization")
                continue
            
            # Create a figure with three panels: original, cleaned, aligned
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Check if there are any valid pixels in the cleaned data
            valid_data = image['cleaned_data'][~np.isnan(image['cleaned_data'])]
            if len(valid_data) > 0:
                # Common colormap scaling based on cleaned data
                vmin, vmax = np.percentile(valid_data, [1, 99])
                
                # Original data
                im0 = axes[0].imshow(image['data'], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0].set_title("Original")
                plt.colorbar(im0, ax=axes[0], label='Flux')
                
                # Cleaned data
                im1 = axes[1].imshow(image['cleaned_data'], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1].set_title("Cleaned")
                plt.colorbar(im1, ax=axes[1], label='Flux')
                
                # Aligned data
                im2 = axes[2].imshow(image['aligned_data'], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                axes[2].set_title("Aligned")
                plt.colorbar(im2, ax=axes[2], label='Flux')
            else:
                # Handle case where image has no valid data
                for j, title in enumerate(["Original", "Cleaned", "Aligned"]):
                    axes[j].imshow(np.ones((10, 10))*np.nan, cmap='viridis')
                    axes[j].set_title(title)
                    axes[j].text(0.5, 0.5, "No valid data", ha='center', va='center', 
                                transform=axes[j].transAxes, fontsize=14)
            
            # Add overall title
            fig.suptitle(f"Preprocessing Steps: {image['filename']}", fontsize=16)
            plt.tight_layout()
            
            # Save the figure
            basename = os.path.splitext(image['filename'])[0]
            output_file = os.path.normpath(os.path.join(output_dir, f"{basename}_preprocessing_steps.png"))
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"    Saved visualization to {output_file}")
        except Exception as e:
            print(f"    Error creating visualization for image {i}: {e}")
    
    # Create a comparison of all aligned images
    if len(images) > 1:
        try:
            print("  Creating comparison of all aligned images...")
            
            # Determine grid size for subplot
            n_cols = min(3, len(images))
            n_rows = (len(images) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            # Plot each aligned image
            for i, image in enumerate(images):
                if i < len(axes) and 'aligned_data' in image:
                    valid_data = image['aligned_data'][~np.isnan(image['aligned_data'])]
                    if len(valid_data) > 0:
                        vmin, vmax = np.percentile(valid_data, [1, 99])
                        im = axes[i].imshow(image['aligned_data'], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                        axes[i].set_title(f"Image {i}: {image['filename']}")
                        plt.colorbar(im, ax=axes[i], label='Flux')
                    else:
                        axes[i].imshow(np.ones((10, 10))*np.nan, cmap='viridis')
                        axes[i].set_title(f"Image {i}: {image['filename']}")
                        axes[i].text(0.5, 0.5, "No valid data", ha='center', va='center', 
                                    transform=axes[i].transAxes, fontsize=14)
            
            # Hide unused subplots
            for i in range(len(images), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            output_file = os.path.normpath(os.path.join(output_dir, "aligned_images_comparison.png"))
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"    Saved comparison to {output_file}")
        except Exception as e:
            print(f"    Error creating image comparison: {e}")
    
    # Create a stacked image visualization
    if len(images) > 1:
        try:
            print("  Creating stacked image visualization...")
            
            # Check if all images have aligned_data
            if not all('aligned_data' in img for img in images):
                print("    Not all images have aligned data, skipping stack visualization")
                return
            
            # Get dimensions that all images have in common
            min_shape = np.min(np.array([img['aligned_data'].shape for img in images]), axis=0)
            
            # Extract valid regions from all images
            valid_data = []
            for img in images:
                # Crop to common dimensions
                cropped = img['aligned_data'][:min_shape[0], :min_shape[1]]
                valid_data.append(cropped)
            
            # Create mean, median and sum stacks
            with np.errstate(invalid='ignore'):  # Suppress warnings about NaNs
                mean_stack = np.nanmean(valid_data, axis=0)
                median_stack = np.nanmedian(valid_data, axis=0)
                sum_stack = np.nansum(valid_data, axis=0)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Function to display stack
            def display_stack(ax, stack, title):
                valid = stack[~np.isnan(stack)]
                if len(valid) > 0:
                    vmin, vmax = np.percentile(valid, [1, 99])
                    im = ax.imshow(stack, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                    ax.set_title(title)
                    return im
                else:
                    ax.imshow(np.ones((10, 10))*np.nan, cmap='viridis')
                    ax.set_title(title)
                    ax.text(0.5, 0.5, "No valid data", ha='center', va='center', 
                            transform=ax.transAxes, fontsize=14)
                    return None
            
            # Display the three stacks
            im0 = display_stack(axes[0], mean_stack, "Mean Stack")
            im1 = display_stack(axes[1], median_stack, "Median Stack")
            im2 = display_stack(axes[2], sum_stack, "Sum Stack")
            
            # Add colorbars where we have valid data
            for im, ax in [(im0, axes[0]), (im1, axes[1]), (im2, axes[2])]:
                if im is not None:
                    plt.colorbar(im, ax=ax)
            
            fig.suptitle(f"Stacked Images (n={len(images)})", fontsize=16)
            plt.tight_layout()
            
            output_file = os.path.normpath(os.path.join(output_dir, "stacked_images.png"))
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"    Saved stacked visualization to {output_file}")
        except Exception as e:
            print(f"    Error creating stacked visualizations: {e}")
            traceback.print_exc()

def align_images_safely(images, reference_idx=0, method='centroid', verbose=True):
    """
    A safer wrapper for the align_images function with appropriate fallbacks
    
    Parameters:
    -----------
    images : list
        List of image dictionaries from fits_loader
    reference_idx : int
        Index of the reference image
    method : str
        Alignment method: 'correlation', 'centroid', or 'wcs'
    verbose : bool
        Whether to print verbose information
        
    Returns:
    --------
    list
        List of aligned image dictionaries
    """
    print(f"Attempting to align images using {method} method...")
    
    # First, create a backup copy of the cleaned data for each image
    # This ensures we still have something if alignment fails completely
    for img in images:
        if 'cleaned_data' in img:
            img['aligned_data'] = img['cleaned_data'].copy()
            img['alignment_shift'] = (0.0, 0.0)
    
    # Define a function to try different alignment methods with timeout
    def try_alignment(method, timeout=180):
        try:
            print(f"  Trying {method} method with {timeout}s timeout...")
            start_time = time.time()
            
            # Wrapped function to call alignment
            def align_wrapper():
                from preprocessing.alignment import align_images as original_align
                return original_align(images, reference_idx, method, verbose)
            
            # Run with timeout
            result = run_with_timeout(align_wrapper, timeout_seconds=timeout)
            
            elapsed = time.time() - start_time
            print(f"  Alignment completed in {elapsed:.1f} seconds")
            return result, True
        except TimeoutError:
            print(f"  Alignment timed out after {timeout} seconds")
            return images, False
        except Exception as e:
            print(f"  Alignment failed with error: {e}")
            return images, False
    
    # Try the requested method first
    aligned_images, success = try_alignment(method)
    
    # If that fails, try centroid method (if it wasn't already)
    if not success and method != 'centroid':
        aligned_images, success = try_alignment('centroid', timeout=120)
    
    # If that also fails, try WCS method (if it wasn't already)
    if not success and method != 'wcs':
        aligned_images, success = try_alignment('wcs', timeout=60)
    
    # If all methods fail, use simple identity alignment
    if not success:
        print("  All alignment methods failed. Using unaligned images.")
        # The backup aligned_data was already created at the start
    
    return aligned_images

def preprocess_sequence(fits_files, output_dir, visualization_dir=None, alignment_method='centroid', verbose=True):
    """
    Preprocess a sequence of FITS files and prepare them for KBO detection
    
    Steps:
    1. Load the FITS files
    2. Subtract background from each image
    3. Clean the images (remove bad pixels)
    4. Align the images to a common reference frame
    5. Save the preprocessed data
    """
    # Normalize paths
    output_dir = os.path.normpath(output_dir)
    if visualization_dir:
        visualization_dir = os.path.normpath(visualization_dir)
        
    if verbose:
        print(f"\n=== Preprocessing {len(fits_files)} FITS files ===\n")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if visualization_dir:
        os.makedirs(visualization_dir, exist_ok=True)
    
    # Load FITS files
    images = []
    for fits_file in fits_files:
        try:
            image = load_fits_file(fits_file, verbose=verbose)
            if image and 'data' in image and image['data'] is not None:
                # Check if the data is usable
                if np.all(np.isnan(image['data'])):
                    print(f"  Warning: All data is NaN in {fits_file}, skipping")
                    continue
                    
                # Memory usage check - skip extremely large images
                data_size_gb = image['data'].nbytes / (1024**3)
                if data_size_gb > 1.0:  # Skip images larger than 1GB
                    print(f"  Warning: Image size is {data_size_gb:.2f} GB for {fits_file}, skipping")
                    continue
                    
                images.append(image)
            else:
                print(f"  Warning: Failed to load valid data from {fits_file}")
        except Exception as e:
            print(f"  Error loading {fits_file}: {e}")
    
    if not images:
        print("No valid FITS files found!")
        return None
    
    if verbose:
        print(f"\nLoaded {len(images)} FITS files successfully")
    
    # Process each image
    for i, image in enumerate(images):
        try:
            if verbose:
                print(f"\n--- Processing image {i+1}/{len(images)}: {image['filename']} ---")
            
            # Subtract background
            t_start = time.time()
            image['cleaned_data'], image['bg_level'], image['bg_noise'] = subtract_background(
                image['data'], verbose=verbose)
            
            if verbose:
                elapsed = time.time() - t_start
                print(f"  Background subtraction completed in {elapsed:.1f} seconds")
            
            # Clean image
            t_start = time.time()
            image['cleaned_data'], image['bad_pixel_mask'] = clean_image(
                image['cleaned_data'], verbose=verbose)
                
            if verbose:
                elapsed = time.time() - t_start
                print(f"  Image cleaning completed in {elapsed:.1f} seconds")
                
            # Add a copy as aligned_data in case alignment fails
            image['aligned_data'] = image['cleaned_data'].copy()
            image['alignment_shift'] = (0.0, 0.0)
        
        except Exception as e:
            print(f"  Error processing image {i+1}: {e}")
            # Create dummy cleaned data from original if available
            if 'data' in image:
                image['cleaned_data'] = image['data'].copy()
                image['aligned_data'] = image['data'].copy()
                image['bg_level'] = 0.0
                image['bg_noise'] = 0.0
                image['alignment_shift'] = (0.0, 0.0)
    
    # Align images
    if len(images) > 1:
        # Only try alignment if we have multiple images
        aligned_images = align_images_safely(images, reference_idx=0, method=alignment_method, verbose=verbose)
    else:
        # For single image, just use as is
        aligned_images = images
    
    # Save preprocessed data
    metadata_file = save_preprocessed_data(aligned_images, output_dir)
    
    # Generate visualizations if requested
    if visualization_dir:
        try:
            visualize_preprocessing(aligned_images, visualization_dir)
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    return metadata_file

def find_field_directories(base_dir):
    """
    Find all subdirectories in the base directory that may contain field data
    
    Parameters:
    -----------
    base_dir : str
        Base directory to search for field directories
    
    Returns:
    --------
    list
        List of field directory paths
    """
    base_dir = os.path.normpath(base_dir)
    
    # Check if the base directory exists
    if not os.path.isdir(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        return []
    
    # Get all subdirectories
    try:
        subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    except Exception as e:
        print(f"Error listing subdirectories in {base_dir}: {e}")
        return []
    
    # Check each subdirectory for FITS files
    field_dirs = []
    for subdir in subdirs:
        fits_files = glob(os.path.join(subdir, "*.fits"))
        if fits_files:
            field_dirs.append(subdir)
    
    return field_dirs

def process_field(fits_dir, output_dir, visualization_dir, field_id=None, alignment_method='centroid', verbose=True, max_files=None):
    """
    Process a single field, either with a specified field_id or using the base directory name
    
    Parameters:
    -----------
    fits_dir : str
        Directory containing FITS files for this field
    output_dir : str
        Base output directory for preprocessed files
    visualization_dir : str
        Base output directory for visualizations
    field_id : str or None
        Field ID to use in output paths
    alignment_method : str
        Method for aligning images
    verbose : bool
        Whether to print verbose output
    max_files : int or None
        Maximum number of files to process
    
    Returns:
    --------
    bool
        True if processing succeeded, False otherwise
    """
    try:
        # Determine field_id if not provided (use directory name)
        if field_id is None:
            field_id = os.path.basename(os.path.normpath(fits_dir))
        
        # Create field-specific output directories
        field_output_dir = os.path.normpath(os.path.join(output_dir, field_id))
        field_vis_dir = os.path.normpath(os.path.join(visualization_dir, field_id)) if visualization_dir else None
        
        # Find FITS files
        fits_files = glob(os.path.normpath(os.path.join(fits_dir, "*.fits")))
        
        if not fits_files:
            print(f"No FITS files found in {fits_dir}")
            print(f"Absolute path: {os.path.abspath(fits_dir)}")
            try:
                # Check if directory exists
                if os.path.isdir(fits_dir):
                    print(f"Directory exists. Contents: {os.listdir(fits_dir)}")
                else:
                    print(f"Directory does not exist: {fits_dir}")
                    parent_dir = os.path.dirname(fits_dir)
                    if os.path.isdir(parent_dir):
                        print(f"Parent directory exists. Contents: {os.listdir(parent_dir)}")
            except Exception as e:
                print(f"Error checking directory: {e}")
            return False
        
        # Limit number of files if requested
        if max_files and len(fits_files) > max_files:
            print(f"Limiting to {max_files} files for processing")
            fits_files = fits_files[:max_files]
        
        print(f"Found {len(fits_files)} FITS files in {fits_dir}")
        
        # Preprocess the files
        metadata_file = preprocess_sequence(
            fits_files, 
            field_output_dir, 
            field_vis_dir, 
            alignment_method=alignment_method,
            verbose=verbose
        )
        
        if metadata_file:
            print(f"\nPreprocessing of field {field_id} complete! Metadata saved to {metadata_file}")
            return True
        else:
            print(f"\nPreprocessing of field {field_id} failed!")
            return False
            
    except Exception as e:
        print(f"Error processing field {field_id}: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Preprocess JWST FITS files for KBO detection")
    parser.add_argument('--fits-dir', required=True, help="Directory containing FITS files")
    parser.add_argument('--field-id', help="Field ID to process (subfolder of fits-dir). If not specified, process all fields.")
    parser.add_argument('--output-dir', default="./data/preprocessed", help="Output directory for preprocessed files")
    parser.add_argument('--visualization-dir', default="./data/visualization/preprocessing", 
                        help="Output directory for visualizations")
    parser.add_argument('--alignment-method', choices=['correlation', 'centroid', 'wcs'], default='centroid',
                        help="Method for aligning images (default: centroid)")
    parser.add_argument('--verbose', '-v', action='store_true', help="Print verbose output")
    parser.add_argument('--all', '-a', action='store_true', help="Process all field directories, including the base directory")
    parser.add_argument('--max-files', type=int, help="Maximum number of files to process per field (for testing)")
    parser.add_argument('--skip-alignment', action='store_true', help="Skip alignment step (useful for problematic datasets)")
    
    args = parser.parse_args()
    
    # If skip_alignment is set, override the method
    if args.skip_alignment:
        print("Alignment will be skipped as requested")
        alignment_method = 'none'
    else:
        alignment_method = args.alignment_method
    
    # Normalize base paths
    fits_dir = os.path.normpath(args.fits_dir)
    output_dir = os.path.normpath(args.output_dir)
    vis_dir = os.path.normpath(args.visualization_dir) if args.visualization_dir else None
    
    # Create base output directories
    os.makedirs(output_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    
    try:
        # Process based on arguments
        if args.field_id:
            # Process a single specified field
            field_fits_dir = os.path.normpath(os.path.join(fits_dir, args.field_id))
            
            # Process the field
            success = process_field(
                field_fits_dir, 
                output_dir, 
                vis_dir, 
                field_id=args.field_id,
                alignment_method=alignment_method,
                verbose=args.verbose,
                max_files=args.max_files
            )
            if not success:
                print(f"Failed to process field: {args.field_id}")
                return
        else:
            # Process all fields - find all subdirectories with FITS files
            field_dirs = find_field_directories(fits_dir)
            
            # Check if we should also process the base directory
            if args.all:
                # Check if base directory has FITS files directly
                base_fits_files = glob(os.path.join(fits_dir, "*.fits"))
                if base_fits_files:
                    print(f"\n=== Processing base directory as a field ===")
                    
                    # Process the field
                    process_field(
                        fits_dir,
                        output_dir,
                        vis_dir,
                        field_id="base",
                        alignment_method=alignment_method,
                        verbose=args.verbose,
                        max_files=args.max_files
                    )
            
            if not field_dirs:
                print(f"No field directories with FITS files found in {fits_dir}")
                return
            
            print(f"Found {len(field_dirs)} field directories to process")
            
            # Process each field
            successful_fields = 0
            for i, field_dir in enumerate(field_dirs):
                field_id = os.path.basename(field_dir)
                print(f"\n=== Processing field {i+1}/{len(field_dirs)}: {field_id} ===")
                
                # Process the field
                success = process_field(
                    field_dir,
                    output_dir,
                    vis_dir,
                    field_id=field_id,
                    alignment_method=alignment_method,
                    verbose=args.verbose,
                    max_files=args.max_files
                )
                
                if success:
                    successful_fields += 1
            
            print(f"\n=== Preprocessing Summary ===")
            print(f"Processed {len(field_dirs)} fields")
            print(f"Successfully preprocessed {successful_fields} fields")
            if successful_fields < len(field_dirs):
                print(f"Failed to preprocess {len(field_dirs) - successful_fields} fields")
    
    except KeyboardInterrupt:
        print("\nPreprocessing interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"Unhandled error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()