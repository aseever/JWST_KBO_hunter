#!/usr/bin/env python3
"""
preprocess.py - Main script for preprocessing JWST FITS files

This script uses the modules in the preprocessing package to:
1. Load and validate FITS files
2. Perform background subtraction and clean bad pixels
3. Align images to a common reference frame
4. Save preprocessed data for KBO detection
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

# Import preprocessing modules
from preprocessing.fits_loader import load_fits_file
from preprocessing.calibration import subtract_background, clean_image
from preprocessing.alignment import align_images, stack_aligned_images

def save_preprocessed_data(images, output_dir):
    """Save preprocessed images for later use"""
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
        output_file = os.path.join(output_dir, f"{basename}_preprocessed.fits")
        
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
                except ValueError:
                    # Skip headers that can't be properly represented in FITS
                    continue
        
        # Add preprocessing information
        hdu.header['BG_LEVEL'] = image['bg_level']
        hdu.header['BG_NOISE'] = image['bg_noise']
        hdu.header['PREPROC'] = True
        hdu.header['ALIGNED'] = 'aligned_data' in image
        
        # Add alignment info if available
        if 'alignment_shift' in image:
            hdu.header['ALIGN_DX'] = image['alignment_shift'][0]
            hdu.header['ALIGN_DY'] = image['alignment_shift'][1]
        
        # Save the FITS file
        hdu.writeto(output_file, overwrite=True)
        print(f"  Saved {output_file}")
        
        # Create a quick preview image
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
        preview_file = os.path.join(output_dir, f"{basename}_preview.png")
        plt.savefig(preview_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Add to metadata
        metadata['images'].append({
            'index': i,
            'filename': image['filename'],
            'output_file': os.path.basename(output_file),
            'preview_file': os.path.basename(preview_file),
            'instrument': image['instrument'],
            'filter': image['filter'],
            'exptime': image['exptime'],
            'bg_level': float(image['bg_level']),
            'bg_noise': float(image['bg_noise']),
            'shape': image['aligned_data'].shape
        })
    
    # Save metadata
    metadata_file = os.path.join(output_dir, "preprocessing_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved metadata to {metadata_file}")
    
    return metadata_file

def visualize_preprocessing(images, output_dir):
    """Create before/after visualizations of the preprocessing steps"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating preprocessing visualizations...")
    
    for i, image in enumerate(images):
        print(f"  Creating visualization for image {i}: {image['filename']}")
        
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
        output_file = os.path.join(output_dir, f"{basename}_preprocessing_steps.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved visualization to {output_file}")
    
    # Create a comparison of all aligned images
    if len(images) > 1:
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
            if i < len(axes):
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
        output_file = os.path.join(output_dir, "aligned_images_comparison.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved comparison to {output_file}")
    
    # Create a stacked image visualization
    if len(images) > 1:
        print("  Creating stacked image visualization...")
        
        # Get aligned data
        aligned_data = [img['aligned_data'] for img in images]
        
        # Calculate stacks
        try:
            # Create mean, median and sum stacks
            mean_stack = np.nanmean(aligned_data, axis=0)
            median_stack = np.nanmedian(aligned_data, axis=0)
            sum_stack = np.nansum(aligned_data, axis=0)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Check if we have any valid data in stacks
            valid_mean = mean_stack[~np.isnan(mean_stack)]
            valid_median = median_stack[~np.isnan(median_stack)]
            valid_sum = sum_stack[~np.isnan(sum_stack)]
            
            # Mean stack
            if len(valid_mean) > 0:
                vmin, vmax = np.percentile(valid_mean, [1, 99])
                im0 = axes[0].imshow(mean_stack, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0].set_title("Mean Stack")
                plt.colorbar(im0, ax=axes[0])
            else:
                axes[0].imshow(np.ones((10, 10))*np.nan, cmap='viridis')
                axes[0].set_title("Mean Stack")
                axes[0].text(0.5, 0.5, "No valid data", ha='center', va='center', 
                            transform=axes[0].transAxes, fontsize=14)
            
            # Median stack
            if len(valid_median) > 0:
                vmin, vmax = np.percentile(valid_median, [1, 99])
                im1 = axes[1].imshow(median_stack, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1].set_title("Median Stack")
                plt.colorbar(im1, ax=axes[1])
            else:
                axes[1].imshow(np.ones((10, 10))*np.nan, cmap='viridis')
                axes[1].set_title("Median Stack")
                axes[1].text(0.5, 0.5, "No valid data", ha='center', va='center', 
                            transform=axes[1].transAxes, fontsize=14)
            
            # Sum stack
            if len(valid_sum) > 0:
                vmin, vmax = np.percentile(valid_sum, [1, 99])
                im2 = axes[2].imshow(sum_stack, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
                axes[2].set_title("Sum Stack")
                plt.colorbar(im2, ax=axes[2])
            else:
                axes[2].imshow(np.ones((10, 10))*np.nan, cmap='viridis')
                axes[2].set_title("Sum Stack")
                axes[2].text(0.5, 0.5, "No valid data", ha='center', va='center', 
                            transform=axes[2].transAxes, fontsize=14)
            
            fig.suptitle(f"Stacked Images (n={len(images)})", fontsize=16)
            plt.tight_layout()
            
            output_file = os.path.join(output_dir, "stacked_images.png")
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved stacked visualization to {output_file}")
        except Exception as e:
            print(f"    Error creating stacked visualizations: {e}")

def preprocess_sequence(fits_files, output_dir, visualization_dir=None, alignment_method='correlation', verbose=True):
    """
    Preprocess a sequence of FITS files and prepare them for KBO detection
    
    Steps:
    1. Load the FITS files
    2. Subtract background from each image
    3. Clean the images (remove bad pixels)
    4. Align the images to a common reference frame
    5. Save the preprocessed data
    """
    if verbose:
        print(f"\n=== Preprocessing {len(fits_files)} FITS files ===\n")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if visualization_dir:
        os.makedirs(visualization_dir, exist_ok=True)
    
    # Load FITS files
    images = []
    for fits_file in fits_files:
        image = load_fits_file(fits_file, verbose=verbose)
        if image:
            images.append(image)
    
    if not images:
        print("No valid FITS files found!")
        return None
    
    if verbose:
        print(f"\nLoaded {len(images)} FITS files successfully")
    
    # Process each image
    for i, image in enumerate(images):
        if verbose:
            print(f"\n--- Processing image {i+1}/{len(images)}: {image['filename']} ---")
        
        # Subtract background
        image['cleaned_data'], image['bg_level'], image['bg_noise'] = subtract_background(
            image['data'], verbose=verbose)
        
        # Clean image
        image['cleaned_data'], image['bad_pixel_mask'] = clean_image(
            image['cleaned_data'], verbose=verbose)
    
    # Align images
    aligned_images = align_images(images, method=alignment_method, verbose=verbose)
    
    # Save preprocessed data
    metadata_file = save_preprocessed_data(aligned_images, output_dir)
    
    # Generate visualizations if requested
    if visualization_dir:
        visualize_preprocessing(aligned_images, visualization_dir)
    
    return metadata_file

def main():
    parser = argparse.ArgumentParser(description="Preprocess JWST FITS files for KBO detection")
    parser.add_argument('--fits-dir', required=True, help="Directory containing FITS files")
    parser.add_argument('--field-id', help="Field ID to process (subfolder of fits-dir)")
    parser.add_argument('--output-dir', default="./data/preprocessed", help="Output directory for preprocessed files")
    parser.add_argument('--visualization-dir', default="./data/visualization/preprocessing", 
                        help="Output directory for visualizations")
    parser.add_argument('--alignment-method', choices=['correlation', 'centroid', 'wcs'], default='correlation',
                        help="Method for aligning images (default: correlation)")
    parser.add_argument('--verbose', '-v', action='store_true', help="Print verbose output")
    
    args = parser.parse_args()
    
    # If field_id is provided, look in that subfolder
    if args.field_id:
        fits_dir = os.path.join(args.fits_dir, args.field_id)
        output_subdir = os.path.join(args.output_dir, args.field_id)
        vis_subdir = os.path.join(args.visualization_dir, args.field_id)
    else:
        fits_dir = args.fits_dir
        output_subdir = args.output_dir
        vis_subdir = args.visualization_dir
    
    # Find FITS files
    fits_files = glob(os.path.join(fits_dir, "*.fits"))
    
    if not fits_files:
        print(f"No FITS files found in {fits_dir}")
        return
    
    print(f"Found {len(fits_files)} FITS files in {fits_dir}")
    
    # Preprocess the files
    metadata_file = preprocess_sequence(
        fits_files, 
        output_subdir, 
        vis_subdir, 
        alignment_method=args.alignment_method,
        verbose=args.verbose
    )
    
    if metadata_file:
        print(f"\nPreprocessing complete! Metadata saved to {metadata_file}")
    else:
        print("\nPreprocessing failed!")

if __name__ == "__main__":
    main()