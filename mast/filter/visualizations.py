"""
mast/filter/visualizations.py - Visualization utilities for KBO filter results

This module provides functions to create visualizations of the filtering results,
including sequence statistics, candidate distributions, and filter effectiveness.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from datetime import datetime

# Set up logger
logger = logging.getLogger('mast_kbo')

def generate_filter_visualizations(results, output_dir=None):
    """
    Generate visualizations of the filtering results
    
    Parameters:
    -----------
    results : dict
        Results from filter_catalog
    output_dir : str or None
        Output directory for visualizations
        
    Returns:
    --------
    list : Paths to generated visualization files
    """
    if not results or 'sequences' not in results:
        logger.error("Invalid results for visualization")
        return []
    
    # Determine output directory
    if output_dir is None:
        if 'source_catalog' in results:
            output_dir = os.path.join(os.path.dirname(results['source_catalog']), 'visualizations')
        else:
            output_dir = 'visualizations'
    
    os.makedirs(output_dir, exist_ok=True)
    
    visualization_files = []
    
    # 1. Sequence length histogram
    if results['sequences']:
        seq_length_file = visualize_sequence_lengths(results['sequences'], output_dir)
        if seq_length_file:
            visualization_files.append(seq_length_file)
    
    # 2. Sequence duration histogram
    if results['sequences']:
        seq_duration_file = visualize_sequence_durations(results['sequences'], output_dir)
        if seq_duration_file:
            visualization_files.append(seq_duration_file)
    
    # 3. Sequence scores
    if results['sequences']:
        seq_score_file = visualize_sequence_scores(results['sequences'], output_dir)
        if seq_score_file:
            visualization_files.append(seq_score_file)
    
    # 4. Filter effectiveness pie chart
    stats = results.get('stats', {})
    if 'initial_observations' in stats and 'filtered_observations' in stats:
        filter_effect_file = visualize_filter_effectiveness(stats, output_dir)
        if filter_effect_file:
            visualization_files.append(filter_effect_file)
    
    # 5. Create combined dashboard (if we have sequences)
    if results['sequences'] and len(results['sequences']) > 0:
        dashboard_file = create_sequence_dashboard(results, output_dir)
        if dashboard_file:
            visualization_files.append(dashboard_file)
            
    return visualization_files

def visualize_sequence_lengths(sequences, output_dir):
    """
    Create histogram of sequence lengths
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
    output_dir : str
        Output directory
        
    Returns:
    --------
    str or None : Path to output file if successful
    """
    try:
        plt.figure(figsize=(10, 6))
        seq_lengths = [seq['num_observations'] for seq in sequences]
        plt.hist(seq_lengths, bins=range(2, max(seq_lengths) + 2), alpha=0.7, 
                color='steelblue', edgecolor='black')
        plt.xlabel('Number of Observations in Sequence')
        plt.ylabel('Count')
        plt.title('Observation Sequence Lengths')
        plt.xticks(range(2, max(seq_lengths) + 1))
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(output_dir, 'sequence_lengths.png')
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"Generated sequence length visualization: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error generating sequence length visualization: {e}")
        return None

def visualize_sequence_durations(sequences, output_dir):
    """
    Create histogram of sequence durations
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
    output_dir : str
        Output directory
        
    Returns:
    --------
    str or None : Path to output file if successful
    """
    try:
        plt.figure(figsize=(10, 6))
        durations = [seq.get('duration_hours', 0) for seq in sequences]
        
        # Calculate optimal bin width based on Freedman-Diaconis rule
        if len(durations) > 1:
            q75, q25 = np.percentile(durations, [75, 25])
            bin_width = 2 * (q75 - q25) * len(durations)**(-1/3)
            if bin_width > 0:
                num_bins = int(np.ceil((max(durations) - min(durations)) / bin_width))
                num_bins = min(max(num_bins, 5), 20)  # Keep between 5 and 20 bins
            else:
                num_bins = 10
        else:
            num_bins = 10
            
        plt.hist(durations, bins=num_bins, alpha=0.7, color='mediumseagreen', edgecolor='black')
        plt.xlabel('Sequence Duration (hours)')
        plt.ylabel('Count')
        plt.title('Observation Sequence Durations')
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(output_dir, 'sequence_durations.png')
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"Generated sequence duration visualization: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error generating sequence duration visualization: {e}")
        return None

def visualize_sequence_scores(sequences, output_dir):
    """
    Create histogram of sequence KBO detection scores
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
    output_dir : str
        Output directory
        
    Returns:
    --------
    str or None : Path to output file if successful
    """
    try:
        plt.figure(figsize=(10, 6))
        scores = [seq.get('kbo_score', 0) for seq in sequences]
        
        # Create custom colormap for the histogram bars based on score ranges
        cmap = LinearSegmentedColormap.from_list(
            'score_cmap', [(0.2, 'lightcoral'), (0.5, 'gold'), (0.8, 'forestgreen')])
        
        # Create bin edges from 0 to 1 in 0.1 increments
        bin_edges = np.linspace(0, 1, 11)
        
        # Create histogram
        n, bins, patches = plt.hist(scores, bins=bin_edges, edgecolor='black', alpha=0.8)
        
        # Color bins based on score value
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        for c, p in zip(bin_centers, patches):
            plt.setp(p, 'facecolor', cmap(c))
        
        plt.xlabel('KBO Detection Score')
        plt.ylabel('Count')
        plt.title('Sequence KBO Detection Scores')
        plt.grid(True, alpha=0.3)
        
        # Add color legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, label='Score Quality')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['Low', 'Fair', 'Medium', 'Good', 'High'])
        
        output_file = os.path.join(output_dir, 'sequence_scores.png')
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"Generated sequence score visualization: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error generating sequence score visualization: {e}")
        return None

def visualize_filter_effectiveness(stats, output_dir):
    """
    Create pie chart of filter effectiveness
    
    Parameters:
    -----------
    stats : dict
        Statistics dictionary from filter results
    output_dir : str
        Output directory
        
    Returns:
    --------
    str or None : Path to output file if successful
    """
    try:
        plt.figure(figsize=(8, 8))
        
        # Create data for pie chart
        initial = stats['initial_observations']
        filtered = stats['filtered_observations']
        removed = initial - filtered
        
        sizes = [filtered, removed]
        labels = [f'Passed Filters\n({filtered} obs)', f'Filtered Out\n({removed} obs)']
        colors = ['mediumseagreen', 'lightgray']
        explode = (0.1, 0)  # Explode the first slice
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Filter Effectiveness')
        
        # Add additional stats as text
        kept_percent = filtered / initial * 100 if initial > 0 else 0
        plt.figtext(0.5, 0.01, 
                   f"Total: {initial} observations | Kept: {kept_percent:.1f}% | Filtered: {100-kept_percent:.1f}%",
                   ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        output_file = os.path.join(output_dir, 'filter_effectiveness.png')
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"Generated filter effectiveness visualization: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error generating filter effectiveness visualization: {e}")
        return None

def create_sequence_dashboard(results, output_dir):
    """
    Create a comprehensive dashboard of sequence properties
    
    Parameters:
    -----------
    results : dict
        Results from filter_catalog
    output_dir : str
        Output directory
        
    Returns:
    --------
    str or None : Path to output file if successful
    """
    try:
        sequences = results['sequences']
        if not sequences:
            return None
            
        # Create a figure with a grid layout
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # 1. Sequence lengths histogram
        ax1 = fig.add_subplot(gs[0, 0])
        seq_lengths = [seq['num_observations'] for seq in sequences]
        ax1.hist(seq_lengths, bins=range(2, max(seq_lengths) + 2), alpha=0.7, 
                color='steelblue', edgecolor='black')
        ax1.set_xlabel('Number of Observations')
        ax1.set_ylabel('Count')
        ax1.set_title('Sequence Lengths')
        ax1.grid(True, alpha=0.3)
        
        # 2. Sequence durations histogram
        ax2 = fig.add_subplot(gs[0, 1])
        durations = [seq.get('duration_hours', 0) for seq in sequences]
        ax2.hist(durations, bins=10, alpha=0.7, color='mediumseagreen', edgecolor='black')
        ax2.set_xlabel('Duration (hours)')
        ax2.set_ylabel('Count')
        ax2.set_title('Sequence Durations')
        ax2.grid(True, alpha=0.3)
        
        # 3. KBO scores histogram
        ax3 = fig.add_subplot(gs[0, 2])
        scores = [seq.get('kbo_score', 0) for seq in sequences]
        
        # Custom colormap
        cmap = LinearSegmentedColormap.from_list(
            'score_cmap', [(0.2, 'lightcoral'), (0.5, 'gold'), (0.8, 'forestgreen')])
        
        # Create histogram
        bin_edges = np.linspace(0, 1, 11)
        n, bins, patches = ax3.hist(scores, bins=bin_edges, edgecolor='black', alpha=0.8)
        
        # Color bins based on score value
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        for c, p in zip(bin_centers, patches):
            plt.setp(p, 'facecolor', cmap(c))
            
        ax3.set_xlabel('KBO Detection Score')
        ax3.set_ylabel('Count')
        ax3.set_title('Sequence Scores')
        ax3.grid(True, alpha=0.3)
        
        # 4. Scatter plot: duration vs. number of observations
        ax4 = fig.add_subplot(gs[1, 0])
        sc = ax4.scatter(
            [seq.get('duration_hours', 0) for seq in sequences],
            [seq['num_observations'] for seq in sequences],
            c=[seq.get('kbo_score', 0) for seq in sequences],
            cmap=cmap,
            s=80,
            alpha=0.8,
            edgecolor='black'
        )
        ax4.set_xlabel('Duration (hours)')
        ax4.set_ylabel('Number of Observations')
        ax4.set_title('Duration vs. Observations')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax4, label='KBO Score')
        
        # 5. Filter effectiveness pie chart
        ax5 = fig.add_subplot(gs[1, 1])
        stats = results.get('stats', {})
        if 'initial_observations' in stats and 'filtered_observations' in stats:
            initial = stats['initial_observations']
            filtered = stats['filtered_observations']
            removed = initial - filtered
            
            sizes = [filtered, removed]
            labels = [f'Passed\n({filtered})', f'Filtered\n({removed})']
            colors = ['mediumseagreen', 'lightgray']
            explode = (0.1, 0)
            
            ax5.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
            ax5.set_title('Filter Effectiveness')
            ax5.axis('equal')
        else:
            ax5.text(0.5, 0.5, "No filter statistics available", 
                    horizontalalignment='center', verticalalignment='center')
            ax5.axis('off')
        
        # 6. Top sequences table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        if sequences:
            # Sort by KBO score
            sorted_seqs = sorted(sequences, key=lambda x: x.get('kbo_score', 0), reverse=True)
            top_seqs = sorted_seqs[:5]  # Top 5 sequences
            
            table_data = []
            for seq in top_seqs:
                table_data.append([
                    seq['num_observations'],
                    f"{seq.get('duration_hours', 0):.1f}",
                    f"{seq.get('kbo_score', 0):.2f}"
                ])
            
            table = ax6.table(
                cellText=table_data,
                colLabels=['# Obs', 'Hours', 'Score'],
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax6.set_title('Top 5 Sequences by Score')
        else:
            ax6.text(0.5, 0.5, "No sequences available", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Add overall title
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        plt.suptitle(f'KBO Detection Filter Results Dashboard\n{timestamp}', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        
        # Save the dashboard
        output_file = os.path.join(output_dir, 'filter_dashboard.png')
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"Generated filter dashboard: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error generating filter dashboard: {e}")
        return None

def visualize_sequence_time_distribution(sequences, output_dir):
    """
    Create timeline visualization of observation sequences
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
    output_dir : str
        Output directory
        
    Returns:
    --------
    str or None : Path to output file if successful
    """
    try:
        # Check if we have enough sequences with time information
        valid_sequences = [seq for seq in sequences if 'start_time' in seq and 'end_time' in seq]
        if len(valid_sequences) < 2:
            logger.warning("Not enough sequences with time information for timeline visualization")
            return None
        
        # Convert times to datetime objects for easier plotting
        from dateutil import parser
        
        for seq in valid_sequences:
            try:
                seq['start_dt'] = parser.parse(seq['start_time'])
                seq['end_dt'] = parser.parse(seq['end_time'])
            except (ValueError, TypeError):
                # Skip sequences with invalid dates
                continue
        
        # Filter again to only keep sequences with valid datetime info
        time_sequences = [seq for seq in valid_sequences if 'start_dt' in seq and 'end_dt' in seq]
        if len(time_sequences) < 2:
            logger.warning("Not enough sequences with valid time information")
            return None
        
        # Sort by start time
        time_sequences.sort(key=lambda x: x['start_dt'])
        
        # Create timeline figure
        plt.figure(figsize=(12, 8))
        
        # Plot each sequence as a horizontal bar
        for i, seq in enumerate(time_sequences):
            # Calculate position
            y_pos = len(time_sequences) - i
            
            # Calculate score-based color
            score = seq.get('kbo_score', 0.5)
            color = plt.cm.RdYlGn(score)
            
            # Plot the sequence as a bar
            plt.barh(
                y_pos,
                (seq['end_dt'] - seq['start_dt']).total_seconds() / 3600,  # Duration in hours
                left=matplotlib.dates.date2num(seq['start_dt']),
                height=0.8,
                color=color,
                alpha=0.8,
                edgecolor='black'
            )
            
            # Add labels for sequence properties
            label = f"{seq['num_observations']} obs"
            plt.text(
                matplotlib.dates.date2num(seq['start_dt']), 
                y_pos, 
                label,
                va='center',
                fontsize=8,
                color='black'
            )
        
        # Format x-axis as dates
        plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()
        
        # Set labels and title
        plt.yticks(range(1, len(time_sequences) + 1), [f"Seq {i+1}" for i in range(len(time_sequences))])
        plt.xlabel('Date/Time')
        plt.ylabel('Sequence ID')
        plt.title('Observation Sequence Timeline')
        plt.grid(True, alpha=0.3)
        
        # Add color legend
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, label='KBO Score')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['Low', 'Fair', 'Medium', 'Good', 'High'])
        
        # Save figure
        output_file = os.path.join(output_dir, 'sequence_timeline.png')
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"Generated sequence timeline visualization: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error generating sequence timeline visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_spatial_distribution(sequences, output_dir):
    """
    Create spatial distribution map of sequences
    
    Parameters:
    -----------
    sequences : list
        List of sequence dictionaries
    output_dir : str
        Output directory
        
    Returns:
    --------
    str or None : Path to output file if successful
    """
    try:
        # Check if we have coordinates for the sequences
        valid_sequences = [seq for seq in sequences 
                          if 'center_ra' in seq and 'center_dec' in seq]
        
        if len(valid_sequences) < 2:
            logger.warning("Not enough sequences with coordinate information")
            return None
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Extract coordinates and properties
        ra = [seq['center_ra'] for seq in valid_sequences]
        dec = [seq['center_dec'] for seq in valid_sequences]
        num_obs = [seq['num_observations'] for seq in valid_sequences]
        scores = [seq.get('kbo_score', 0.5) for seq in valid_sequences]
        
        # Handle RA wrap-around (0/360 boundary)
        for i in range(len(ra)):
            if min(ra) < 180 and max(ra) > 270:
                # We have coordinates spanning the 0/360 boundary
                if ra[i] > 180:
                    ra[i] -= 360  # Convert to negative for better visualization
        
        # Create scatter plot
        sc = plt.scatter(
            ra, dec,
            c=scores,
            s=[n * 25 for n in num_obs],  # Size based on number of observations
            cmap='RdYlGn',
            alpha=0.8,
            edgecolor='black'
        )
        
        # Plot the ecliptic plane as a reference
        try:
            from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
            import astropy.units as u
            
            # Generate points along the ecliptic
            ecliptic_lon = np.linspace(0, 360, 360) * u.deg
            ecliptic_lat = np.zeros(360) * u.deg
            
            # Convert to equatorial coordinates
            ecliptic_coords = SkyCoord(
                lon=ecliptic_lon,
                lat=ecliptic_lat,
                frame=GeocentricTrueEcliptic
            )
            
            ecliptic_ra = ecliptic_coords.icrs.ra.degree
            ecliptic_dec = ecliptic_coords.icrs.dec.degree
            
            # Handle RA wrap-around
            for i in range(len(ecliptic_ra)):
                if min(ra) < 180 and max(ra) > 270:
                    if ecliptic_ra[i] > 180:
                        ecliptic_ra[i] -= 360
            
            # Plot ecliptic plane
            plt.plot(
                ecliptic_ra, 
                ecliptic_dec, 
                'k--', 
                alpha=0.5, 
                label='Ecliptic Plane'
            )
            
        except Exception as e:
            logger.warning(f"Could not plot ecliptic plane: {e}")
        
        # Add colorbar and legend
        cbar = plt.colorbar(sc, label='KBO Score')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['Low', 'Fair', 'Medium', 'Good', 'High'])
        
        # Add legend for sizes
        for n in sorted(set(num_obs)):
            plt.scatter([], [], c='grey', alpha=0.8, s=n*25, 
                       edgecolor='black', label=f'{n} observations')
        plt.legend(scatterpoints=1, frameon=True, labelspacing=1)
        
        # Set labels and title
        plt.xlabel('Right Ascension (degrees)')
        plt.ylabel('Declination (degrees)')
        plt.title('Spatial Distribution of Observation Sequences')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        output_file = os.path.join(output_dir, 'sequence_spatial_distribution.png')
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        logger.info(f"Generated spatial distribution visualization: {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error generating spatial distribution visualization: {e}")
        return None