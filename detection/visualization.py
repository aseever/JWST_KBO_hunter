"""
visualization.py - Create visualizations for KBO candidates

This module handles the creation of visualizations for KBO candidates,
showing both the raw images and stacked results.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from detection.shift_stack import stack_images

def visualize_candidates(images, candidates, output_dir, time_span_hours=None, 
                        plate_scale=None, verbose=True):
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
    time_span_hours : float or None
        Time span of the observation in hours
    plate_scale : float or None
        Plate scale in arcseconds per pixel
    verbose : bool
        Whether to print verbose information
    """
    # Normalize output path
    output_dir = os.path.normpath(output_dir)
    
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
        score = candidate.get('score', 0.0)
        
        # Get physical parameters if available
        motion_arcsec_per_hour = candidate.get('motion_arcsec_per_hour', None)
        approx_distance_au = candidate.get('approx_distance_au', None)
        
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
        
        # Create title with physical information if available
        title = f"Candidate {i+1}: Score {score:.2f}, Motion {speed:.1f} pixels at {angle:.1f}°"
        
        if motion_arcsec_per_hour is not None:
            title += f"\nMotion rate: {motion_arcsec_per_hour:.2f} arcsec/hour"
            
        if approx_distance_au is not None:
            title += f", Est. distance: {approx_distance_au:.1f} AU"
            
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f"candidate_{i+1:02d}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
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
                score = candidate.get('score', 0.0)
                
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
                
                # Add motion vector info
                dx, dy = candidate['motion_vector']
                speed = math.sqrt(dx*dx + dy*dy) * len(shifts)
                
                # Get physical parameters if available
                motion_arcsec = candidate.get('motion_arcsec_per_hour', None)
                dist_au = candidate.get('approx_distance_au', None)
                
                # Create title with important information
                title = f"#{i+1}: Score {score:.2f}"
                if motion_arcsec is not None:
                    title += f"\n{motion_arcsec:.1f} ″/hr"
                if dist_au is not None and dist_au > 0:
                    title += f" ({dist_au:.0f} AU)"
                
                axes[i].set_title(title)
                
                # Remove axis ticks
                axes[i].set_xticks([])
                axes[i].set_yticks([])
        
        # Hide unused subplots
        for i in range(num_candidates, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("Top KBO Candidates", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, "candidate_summary.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        if verbose:
            print(f"    Saved summary to {output_file}")

def create_diagnostic_plots(motion_range, filtered_candidates, output_dir):
    """
    Create diagnostic plots for KBO detection results
    
    Parameters:
    -----------
    motion_range : dict
        Dictionary of KBO motion ranges
    filtered_candidates : list
        List of filtered candidates
    output_dir : str
        Output directory for visualizations
    """
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Motion vector distribution
    if filtered_candidates:
        plt.figure(figsize=(10, 8))
        
        # Extract motion vectors from candidates
        dx_values = [c['motion_vector'][0] for c in filtered_candidates]
        dy_values = [c['motion_vector'][1] for c in filtered_candidates]
        scores = [c['score'] for c in filtered_candidates]
        
        # Create scatter plot with colors based on score
        sc = plt.scatter(dx_values, dy_values, c=scores, cmap='viridis', 
                       alpha=0.7, s=100, vmin=0.5, vmax=1.0)
        
        # Add colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label('Candidate Score')
        
        # Add axis labels
        plt.xlabel('Motion in X (pixels per frame)')
        plt.ylabel('Motion in Y (pixels per frame)')
        plt.title('Motion Vectors of KBO Candidates')
        
        # Add expected motion ranges
        for dist in [30, 40, 50, 100]:
            if dist in motion_range['ranges']:
                # Draw a circle representing expected motion at this distance
                motion_pixels = motion_range['ranges'][dist]['total_motion_pixels']
                circle = plt.Circle((0, 0), motion_pixels/10, fill=False, 
                                  linestyle='--', alpha=0.5, label=f"{dist} AU")
                plt.gca().add_patch(circle)
        
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, "motion_vector_distribution.png")
        plt.savefig(output_file, dpi=150)
        plt.close()
    
    # Plot 2: Distance vs. Motion Rate
    if filtered_candidates and any('motion_arcsec_per_hour' in c for c in filtered_candidates):
        plt.figure(figsize=(10, 6))
        
        # Extract data
        distances = [c.get('approx_distance_au', 0) for c in filtered_candidates 
                    if c.get('approx_distance_au', 0) > 0]
        motion_rates = [c.get('motion_arcsec_per_hour', 0) for c in filtered_candidates 
                       if c.get('motion_arcsec_per_hour', 0) > 0 and c.get('approx_distance_au', 0) > 0]
        scores = [c['score'] for c in filtered_candidates 
                if c.get('motion_arcsec_per_hour', 0) > 0 and c.get('approx_distance_au', 0) > 0]
        
        if distances and motion_rates:
            # Create scatter plot
            sc = plt.scatter(distances, motion_rates, c=scores, cmap='viridis', 
                           alpha=0.7, s=100, vmin=0.5, vmax=1.0)
            
            # Add colorbar
            cbar = plt.colorbar(sc)
            cbar.set_label('Candidate Score')
            
            # Add axis labels
            plt.xlabel('Estimated Distance (AU)')
            plt.ylabel('Motion Rate (arcsec/hour)')
            plt.title('KBO Candidates: Distance vs. Motion Rate')
            
            # Add theoretical curve
            x = np.linspace(20, 150, 100)
            y = 4.74 / x  # Simplified formula based on orbital mechanics
            plt.plot(x, y, 'k--', alpha=0.5, label='Theoretical')
            
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, "distance_vs_motion.png")
            plt.savefig(output_file, dpi=150)
            plt.close()