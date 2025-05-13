#!/usr/bin/env python3
"""
ecliptic_survey.py - Systematic KBO survey along the ecliptic

This utility manages a systematic survey of the ecliptic plane for KBO detection,
generating search boxes, tracking progress, and coordinating with kbo_hunt.py.

The tool automates the survey workflow, prioritizing regions of interest and
tracking completed areas to ensure comprehensive coverage of the ecliptic.
"""

import os
import sys
import json
import argparse
import subprocess
import logging
import traceback
from datetime import datetime
import numpy as np
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
import astropy.units as u

# Setup logging
os.makedirs('data', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/ecliptic_survey.log")
    ]
)
logger = logging.getLogger('ecliptic_survey')

# Constants
DEFAULT_SURVEY_FILE = "config/ecliptic_survey_state.json"
DEFAULT_CONFIG_FILE = "config/coordinates.txt"
DEFAULT_RESULTS_DIR = "data/ecliptic_survey_results"
DEFAULT_OVERLAP = 0.25  # 25% overlap between boxes

# Priority regions along the ecliptic (RA hours)
# These are regions of particular interest
PRIORITY_ZONES = [
    (2.0, 4.0),    # Opposition in May 2025
    (12.0, 14.0),  # Away from galactic plane
    (18.5, 20.5)   # Known KBO-rich region
]

# KBO Hunt command configuration templates
DEFAULT_PIPELINE_ARGS = [
    "--ecliptic-priority",
    "--visualize",
    "--analyze"
]

def format_ra(ra_hours):
    """Format RA in hours to HHhMMm00s format"""
    ra_h = int(ra_hours)
    ra_m = int((ra_hours % 1) * 60)
    return f"{ra_h:02d}h{ra_m:02d}m00s"

def format_dec(dec_deg):
    """Format DEC in degrees to +/-DDd00m00s format"""
    dec_sign = '+' if dec_deg >= 0 else ''
    dec_d = int(abs(dec_deg))
    dec_m = int((abs(dec_deg) % 1) * 60)
    return f"{dec_sign}{dec_deg:.2f}d00m00s"

def parse_ra(ra_str):
    """Parse RA from string like '00h30m00s' to hours (float)"""
    parts = ra_str.lower().replace('h', ' ').replace('m', ' ').replace('s', '').split()
    if len(parts) >= 3:
        return float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600
    elif len(parts) == 2:
        return float(parts[0]) + float(parts[1])/60
    else:
        return float(parts[0])

def parse_dec(dec_str):
    """Parse DEC from string like '-03d00m00s' to degrees (float)"""
    parts = dec_str.lower().replace('d', ' ').replace('m', ' ').replace('s', '').split()
    sign = -1 if parts[0].startswith('-') else 1
    
    if len(parts) >= 3:
        return sign * (abs(float(parts[0])) + float(parts[1])/60 + float(parts[2])/3600)
    elif len(parts) == 2:
        return sign * (abs(float(parts[0])) + float(parts[1])/60)
    else:
        return float(parts[0])

def generate_ecliptic_grid(
    ra_start=0.5,       # Starting RA in hours
    ra_end=24.0,        # Ending RA in hours
    box_width=2.0,      # Width of each box in hours
    box_height=6.0,     # Height of each box in degrees
    overlap=0.25,       # Overlap between boxes (fraction)
    priority_zones=None # Optional high-priority RA ranges
):
    """
    Generate a grid of search boxes along the ecliptic plane
    
    Parameters:
    -----------
    ra_start : float
        Starting RA in hours
    ra_end : float
        Ending RA in hours
    box_width : float
        Width of each box in hours
    box_height : float
        Height of each box in degrees
    overlap : float
        Overlap between boxes (fraction)
    priority_zones : list or None
        List of (start, end) RA hour ranges for priority zones
        
    Returns:
    --------
    list : List of search box dictionaries
    """
    # Calculate step size with overlap
    ra_step = box_width * (1 - overlap)
    
    # Generate RA centers
    ra_centers = np.arange(ra_start, ra_end, ra_step)
    
    search_boxes = []
    
    for ra_center in ra_centers:
        # Calculate box bounds
        ra_min = (ra_center - box_width/2) % 24
        ra_max = (ra_center + box_width/2) % 24
        
        # Convert center RA to ecliptic coordinates to find ecliptic plane
        center_eq = SkyCoord(ra=ra_center*15*u.deg, dec=0*u.deg)
        center_ecl = center_eq.transform_to(GeocentricTrueEcliptic)
        
        # Find equatorial coordinates of ecliptic plane at this RA
        ecliptic_point = SkyCoord(
            lon=center_ecl.lon,
            lat=0*u.deg,
            frame=GeocentricTrueEcliptic
        )
        ecliptic_eq = ecliptic_point.transform_to('icrs')
        dec_center = ecliptic_eq.dec.degree
        
        # Create the search box, centered on ecliptic
        box = {
            'box_id': f"RA{ra_min:.1f}-{ra_max:.1f}_DEC{dec_center-box_height/2:.1f}-{dec_center+box_height/2:.1f}",
            'ra_min': format_ra(ra_min),
            'ra_max': format_ra(ra_max),
            'dec_min': format_dec(dec_center-box_height/2),
            'dec_max': format_dec(dec_center+box_height/2),
            'ra_center': ra_center,
            'dec_center': dec_center,
            'priority': 1,  # Default priority
            'status': 'pending'
        }
        
        # Check if in priority zone
        if priority_zones:
            for zone in priority_zones:
                if zone[0] <= ra_center <= zone[1]:
                    box['priority'] = 2  # Higher priority
        
        search_boxes.append(box)
    
    return search_boxes

def create_survey_file(filename=DEFAULT_SURVEY_FILE, overwrite=False):
    """
    Create a new survey state file
    
    Parameters:
    -----------
    filename : str
        Path to the survey file
    overwrite : bool
        Whether to overwrite an existing file
        
    Returns:
    --------
    dict : The created survey state
    """
    if os.path.exists(filename) and not overwrite:
        logger.error(f"Survey file {filename} already exists. Use --force to overwrite.")
        return None
    
    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Generate search boxes along the ecliptic
    boxes = generate_ecliptic_grid(
        ra_start=0.5,
        ra_end=24.0,
        box_width=2.0,
        box_height=6.0,
        overlap=DEFAULT_OVERLAP,
        priority_zones=PRIORITY_ZONES
    )
    
    # Create survey state
    survey_state = {
        'survey_name': f"KBO_Ecliptic_Survey_{datetime.now().strftime('%Y%m%d')}",
        'start_date': datetime.now().strftime('%Y-%m-%d'),
        'boxes_searched': [],
        'next_boxes': boxes,
        'findings': [],
        'created_with': f"ecliptic_survey.py (version 0.2.0)"
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(survey_state, f, indent=2)
    
    logger.info(f"Created new survey state with {len(boxes)} search boxes: {filename}")
    return survey_state

def load_survey_file(filename=DEFAULT_SURVEY_FILE):
    """
    Load an existing survey state file
    
    Parameters:
    -----------
    filename : str
        Path to the survey file
        
    Returns:
    --------
    dict or None : The loaded survey state or None if error
    """
    if not os.path.exists(filename):
        logger.error(f"Survey file {filename} not found. Use --create to create a new one.")
        return None
    
    try:
        with open(filename, 'r') as f:
            survey_state = json.load(f)
        
        logger.info(f"Loaded survey state from {filename}")
        
        # Print quick summary
        boxes_total = len(survey_state['boxes_searched']) + len(survey_state['next_boxes'])
        boxes_done = len(survey_state['boxes_searched'])
        findings = len(survey_state['findings'])
        
        logger.info(f"  Progress: {boxes_done}/{boxes_total} boxes searched ({boxes_done/boxes_total*100:.1f}%)")
        logger.info(f"  Findings: {findings} potential KBO candidates")
        
        return survey_state
    
    except json.JSONDecodeError:
        logger.error(f"Error decoding survey file {filename}. File may be corrupted.")
        return None
    except Exception as e:
        logger.error(f"Error loading survey file {filename}: {e}")
        return None

def update_coordinates_file(box, config_file=DEFAULT_CONFIG_FILE):
    """
    Update the coordinates.txt file with the next search box
    
    Parameters:
    -----------
    box : dict
        Search box dictionary
    config_file : str
        Path to the coordinates.txt file
        
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # Create the config file content
        config_content = f"""# Coordinates for ecliptic survey search box: {box['box_id']}
# Generated by ecliptic_survey.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
RA_MIN = {box['ra_min']}
RA_MAX = {box['ra_max']}
DEC_MIN = {box['dec_min']}
DEC_MAX = {box['dec_max']}"""
        
        # Write to file
        with open(config_file, 'w') as f:
            f.write(config_content)
            
        logger.info(f"Updated coordinates file: {config_file}")
        logger.info(f"  Search box: {box['box_id']}")
        logger.info(f"  RA: {box['ra_min']} to {box['ra_max']}")
        logger.info(f"  DEC: {box['dec_min']} to {box['dec_max']}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating coordinates file: {e}")
        return False

def run_kbo_hunt(box, args=None, results_dir=None):
    """
    Run the KBO hunt pipeline on a search box using the refactored version
    
    Parameters:
    -----------
    box : dict
        Search box dictionary
    args : list or None
        Additional arguments to pass to kbo_hunt.py
    results_dir : str or None
        Directory to store results
        
    Returns:
    --------
    dict : Results from the pipeline
    """
    try:
        # Set up output directory specific to this box
        box_dir = box['box_id']
        if results_dir:
            output_dir = os.path.join(results_dir, box_dir)
        else:
            output_dir = os.path.join(DEFAULT_RESULTS_DIR, box_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct the command
        cmd = [
            sys.executable, 
            "kbo_hunt.py", 
            "pipeline", 
            "--config", DEFAULT_CONFIG_FILE,
            "--output-dir", output_dir
        ]
        
        # Add any additional arguments
        if args:
            cmd.extend(args)
        
        # Run the command
        logger.info(f"Running KBO hunt pipeline for box {box['box_id']}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Run the process with real-time output forwarding
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout to keep log ordering
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Capture and forward output in real time
        output_lines = []
        
        # Process output in real time
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            line = line.rstrip()
            print(f"KBO_HUNT: {line}")  # Print to console
            output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Combine captured output
        output = '\n'.join(output_lines)
        
        if return_code != 0:
            logger.error(f"Pipeline failed with return code {return_code}")
            return {
                'success': False,
                'error': "Pipeline execution failed",
                'output': output,
                'output_dir': output_dir,
                'box_id': box['box_id']
            }
        
        logger.info(f"Pipeline completed successfully")
        
        # Find all result files
        result_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.json'):
                    result_files.append(os.path.join(root, file))
        
        # Look for visualization files
        viz_files = []
        viz_dir = os.path.join(output_dir, "visualizations")
        if os.path.exists(viz_dir):
            for file in os.listdir(viz_dir):
                if file.endswith(('.png', '.jpg', '.svg')):
                    viz_files.append(os.path.join(viz_dir, file))
        
        # Get pipeline summary
        pipeline_summary = None
        for file in result_files:
            if "pipeline_summary_" in file:
                try:
                    with open(file, 'r') as f:
                        pipeline_summary = json.load(f)
                    break
                except:
                    pass
        
        # Get sequenced data
        sequences = []
        for file in result_files:
            if "sequences_" in file or "combined_results_" in file:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        if 'sequences' in data:
                            sequences = data['sequences']
                        elif 'summary' in data and 'sequence_details' in data['summary']:
                            sequences = data['summary']['sequence_details']
                    break
                except:
                    pass
        
        # Build result summary
        return {
            'success': True,
            'output': output,
            'result_files': result_files,
            'visualization_files': viz_files,
            'output_dir': output_dir,
            'box_id': box['box_id'],
            'pipeline_summary': pipeline_summary,
            'sequences': sequences,
            'num_sequences': len(sequences)
        }
        
    except Exception as e:
        logger.error(f"Error running KBO hunt pipeline: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'box_id': box['box_id']
        }

def process_pipeline_results(box, pipeline_result):
    """
    Process the results from the KBO hunt pipeline
    
    Parameters:
    -----------
    box : dict
        Search box dictionary
    pipeline_result : dict
        Results from the pipeline
        
    Returns:
    --------
    dict : Updated box with result information
    """
    if not pipeline_result['success']:
        box['status'] = 'failed'
        box['error'] = pipeline_result.get('error', 'Unknown error')
        return box
    
    # Update box with basic result information
    box['status'] = 'completed'
    box['search_date'] = datetime.now().strftime('%Y-%m-%d')
    box['output_dir'] = pipeline_result.get('output_dir')
    box['result_files'] = pipeline_result.get('result_files', [])
    
    # Check if there are sequences
    sequences = pipeline_result.get('sequences', [])
    box['sequences_found'] = len(sequences)
    
    # Process findings if we have sequences
    if sequences:
        box['has_findings'] = True
        findings = []
        
        for seq in sequences:
            finding = {
                'box_id': box['box_id'],
                'num_observations': seq.get('num_observations', 0),
                'duration_hours': seq.get('duration_hours', 0),
                'center_ra': seq.get('center_ra', box['ra_center']),
                'center_dec': seq.get('center_dec', box['dec_center']),
                'kbo_score': seq.get('kbo_score', 0),
                'discovery_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Add motion information if available
            if 'expected_motion_arcsec' in seq:
                finding['expected_motion_arcsec'] = seq['expected_motion_arcsec']
            
            if 'approx_distance_au' in seq:
                finding['approx_distance_au'] = seq['approx_distance_au']
            
            findings.append(finding)
        
        box['findings'] = findings
    else:
        box['has_findings'] = False
    
    # Add visualization files 
    visualization_files = pipeline_result.get('visualization_files', [])
    if visualization_files:
        box['visualization_files'] = visualization_files
    
    return box

def update_survey_state(survey_state, box):
    """
    Update the survey state with results from a search box
    
    Parameters:
    -----------
    survey_state : dict
        Current survey state
    box : dict
        Processed search box
        
    Returns:
    --------
    dict : Updated survey state
    """
    # Remove the box from next_boxes
    survey_state['next_boxes'] = [b for b in survey_state['next_boxes'] 
                                  if b['box_id'] != box['box_id']]
    
    # Add the box to boxes_searched
    survey_state['boxes_searched'].append(box)
    
    # Add any findings
    if box.get('has_findings', False) and 'findings' in box:
        survey_state['findings'].extend(box['findings'])
    
    # Update survey stats
    boxes_total = len(survey_state['boxes_searched']) + len(survey_state['next_boxes'])
    boxes_done = len(survey_state['boxes_searched'])
    findings = len(survey_state['findings'])
    
    logger.info(f"Updated survey state:")
    logger.info(f"  Progress: {boxes_done}/{boxes_total} boxes searched ({boxes_done/boxes_total*100:.1f}%)")
    logger.info(f"  Findings: {findings} potential KBO candidates")
    
    # Update last modified timestamp
    survey_state['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return survey_state

def save_survey_state(survey_state, filename=DEFAULT_SURVEY_FILE):
    """
    Save the survey state to file
    
    Parameters:
    -----------
    survey_state : dict
        Survey state to save
    filename : str
        Path to the survey file
        
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(survey_state, f, indent=2)
        
        logger.info(f"Saved updated survey state to {filename}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving survey state: {e}")
        return False

def get_next_box(survey_state):
    """
    Get the next search box to process based on priority
    
    Parameters:
    -----------
    survey_state : dict
        Current survey state
        
    Returns:
    --------
    dict or None : Next box to process, or None if none available
    """
    if not survey_state['next_boxes']:
        logger.info("No more search boxes available")
        return None
    
    # Sort by priority (higher first)
    sorted_boxes = sorted(survey_state['next_boxes'], 
                          key=lambda x: x.get('priority', 1), 
                          reverse=True)
    
    # Return the highest priority box
    next_box = sorted_boxes[0]
    logger.info(f"Selected next box: {next_box['box_id']} (priority: {next_box.get('priority', 1)})")
    
    return next_box

def run_survey_iteration(survey_file=DEFAULT_SURVEY_FILE, args=None, results_dir=None):
    """
    Run a single iteration of the survey
    
    Parameters:
    -----------
    survey_file : str
        Path to the survey file
    args : list or None
        Additional arguments to pass to kbo_hunt.py
    results_dir : str or None
        Directory to store results
        
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    # Load the survey state
    survey_state = load_survey_file(survey_file)
    if not survey_state:
        return False
    
    # Get the next box to process
    next_box = get_next_box(survey_state)
    if not next_box:
        logger.info("Survey complete!")
        return True
    
    # Update the coordinates file
    if not update_coordinates_file(next_box):
        return False
    
    # Set up pipeline arguments
    if args is None:
        args = DEFAULT_PIPELINE_ARGS.copy()
    
    # Run the KBO hunt pipeline
    pipeline_result = run_kbo_hunt(next_box, args, results_dir)
    
    # Process the results
    processed_box = process_pipeline_results(next_box, pipeline_result)
    
    # Update the survey state
    updated_state = update_survey_state(survey_state, processed_box)
    
    # Save the updated state
    if not save_survey_state(updated_state, survey_file):
        return False
    
    logger.info("Survey iteration completed successfully")
    return True

def print_survey_summary(survey_state):
    """
    Print a summary of the survey progress
    
    Parameters:
    -----------
    survey_state : dict
        Current survey state
    """
    boxes_total = len(survey_state['boxes_searched']) + len(survey_state['next_boxes'])
    boxes_done = len(survey_state['boxes_searched'])
    findings = len(survey_state['findings'])
    
    print("\n=== KBO Ecliptic Survey Summary ===")
    print(f"Survey: {survey_state['survey_name']}")
    print(f"Started: {survey_state['start_date']}")
    print(f"Last updated: {survey_state.get('last_updated', 'Unknown')}")
    print(f"Progress: {boxes_done}/{boxes_total} boxes searched ({boxes_done/boxes_total*100:.1f}%)")
    print(f"Findings: {findings} potential KBO candidates")
    
    if survey_state['boxes_searched']:
        print("\nRecently searched boxes:")
        for box in sorted(survey_state['boxes_searched'][-5:], 
                         key=lambda x: x.get('search_date', ''), reverse=True):
            status = box.get('status', 'unknown')
            seqs = box.get('sequences_found', 0)
            date = box.get('search_date', 'unknown')
            
            print(f"  {box['box_id']}: {status}, {seqs} sequences ({date})")
    
    if survey_state['findings']:
        print("\nTop KBO candidates (by score):")
        # Sort findings by score
        top_findings = sorted(survey_state['findings'], 
                             key=lambda x: x.get('kbo_score', 0), 
                             reverse=True)[:5]
        
        for i, finding in enumerate(top_findings, 1):
            box_id = finding.get('box_id', 'unknown')
            num_obs = finding.get('num_observations', 0)
            duration = finding.get('duration_hours', 0)
            score = finding.get('kbo_score', 0)
            date = finding.get('discovery_date', 'unknown')
            
            print(f"  {i}. {box_id}: {num_obs} obs over {duration:.1f} hrs, score: {score:.2f} ({date})")
        
        if len(survey_state['findings']) > 5:
            print(f"  ... and {len(survey_state['findings'])-5} more")
    
    if survey_state['next_boxes']:
        next_box = get_next_box(survey_state)
        if next_box:
            print("\nNext search box:")
            print(f"  {next_box['box_id']} (priority: {next_box.get('priority', 1)})")
            print(f"  RA: {next_box['ra_min']} to {next_box['ra_max']}")
            print(f"  DEC: {next_box['dec_min']} to {next_box['dec_max']}")
    
    print("\nTo continue the survey, run:")
    print(f"  python ecliptic_survey.py --run")
    print()

def generate_survey_report(survey_state, output_file=None):
    """
    Generate a comprehensive report of the survey findings
    
    Parameters:
    -----------
    survey_state : dict
        Current survey state
    output_file : str or None
        Path to output report file
        
    Returns:
    --------
    str : Path to the generated report file
    """
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"data/ecliptic_survey_report_{timestamp}.md"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    boxes_total = len(survey_state['boxes_searched']) + len(survey_state['next_boxes'])
    boxes_done = len(survey_state['boxes_searched'])
    findings = len(survey_state['findings'])
    
    # Generate report content
    report = f"""# Ecliptic Survey Report
## {survey_state['survey_name']}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Survey started: {survey_state['start_date']}
- Last updated: {survey_state.get('last_updated', 'Unknown')}
- Progress: {boxes_done}/{boxes_total} boxes searched ({boxes_done/boxes_total*100:.1f}%)
- Findings: {findings} potential KBO candidates

## Top KBO Candidates

| Rank | Box ID | Num Obs | Duration (hrs) | KBO Score | Est. Distance (AU) | Discovery Date |
|------|--------|---------|----------------|-----------|-------------------|----------------|
"""
    
    # Sort findings by score
    sorted_findings = sorted(survey_state['findings'], 
                            key=lambda x: x.get('kbo_score', 0), 
                            reverse=True)
    
    # Add top candidates to report
    for i, finding in enumerate(sorted_findings[:20], 1):
        box_id = finding.get('box_id', 'unknown')
        num_obs = finding.get('num_observations', 0)
        duration = finding.get('duration_hours', 0)
        score = finding.get('kbo_score', 0)
        distance = finding.get('approx_distance_au', 'N/A')
        date = finding.get('discovery_date', 'unknown')
        
        report += f"| {i} | {box_id} | {num_obs} | {duration:.1f} | {score:.2f} | {distance} | {date} |\n"
    
    # Add search coverage
    report += f"""
## Search Coverage

| Status | Count | Percentage |
|--------|-------|------------|
| Completed | {boxes_done} | {boxes_done/boxes_total*100:.1f}% |
| Pending | {len(survey_state['next_boxes'])} | {len(survey_state['next_boxes'])/boxes_total*100:.1f}% |
| Total | {boxes_total} | 100% |

## Visualization

To visualize the survey coverage, you can use the following command:
```
python ecliptic_survey.py --visualize
```

## Next Steps

To continue the survey, run:
```
python ecliptic_survey.py --run
```

To analyze specific findings, check the result directories for each box.
"""
    
    # Write report to file
    with open(output_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Generated survey report: {output_file}")
    return output_file

def visualize_survey_progress(survey_state, output_file=None):
    """
    Visualize the survey progress with matplotlib
    
    Parameters:
    -----------
    survey_state : dict
        Current survey state
    output_file : str or None
        Path to output visualization file
        
    Returns:
    --------
    str : Path to the generated visualization file
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"data/ecliptic_survey_progress_{timestamp}.png"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Plot ecliptic plane
        ecliptic_lons = np.linspace(0, 360, 360)
        ecliptic_lats = np.zeros_like(ecliptic_lons)
        
        # Convert to equatorial coordinates
        ecliptic_coords = SkyCoord(
            lon=ecliptic_lons * u.deg,
            lat=ecliptic_lats * u.deg,
            frame=GeocentricTrueEcliptic
        )
        
        eq_coords = ecliptic_coords.transform_to('icrs')
        
        # Plot the ecliptic line
        plt.plot(eq_coords.ra.hour, eq_coords.dec.deg, 'k--', alpha=0.5, label='Ecliptic')
        
        # Plot completed boxes
        for box in survey_state['boxes_searched']:
            ra_min = parse_ra(box['ra_min'])
            ra_max = parse_ra(box['ra_max'])
            dec_min = parse_dec(box['dec_min'])
            dec_max = parse_dec(box['dec_max'])
            
            # Handle RA wraparound
            if ra_min > ra_max:
                ra_max += 24
            
            width = ra_max - ra_min
            height = dec_max - dec_min
            
            if box.get('status') == 'completed':
                color = 'green' if box.get('has_findings', False) else 'lightgreen'
                alpha = 0.7 if box.get('has_findings', False) else 0.4
            else:
                color = 'red'  # Failed
                alpha = 0.4
            
            rect = Rectangle(
                (ra_min, dec_min), width, height,
                edgecolor='black',
                facecolor=color,
                alpha=alpha,
                label='_nolegend_'
            )
            plt.gca().add_patch(rect)
            
            # Add a marker for findings
            if box.get('has_findings', False):
                for finding in box.get('findings', []):
                    ra = finding.get('center_ra')
                    dec = finding.get('center_dec')
                    score = finding.get('kbo_score', 0)
                    size = 50 * score + 20  # Scale with score
                    
                    if ra is not None and dec is not None:
                        # Convert RA from degrees to hours if needed
                        if ra > 24:
                            ra /= 15
                            
                        plt.scatter(
                            ra, dec,
                            marker='*',
                            s=size,
                            color='yellow',
                            edgecolor='black',
                            zorder=10,
                            label='_nolegend_'
                        )
        
        # Plot pending boxes
        for box in survey_state['next_boxes']:
            ra_min = parse_ra(box['ra_min'])
            ra_max = parse_ra(box['ra_max'])
            dec_min = parse_dec(box['dec_min'])
            dec_max = parse_dec(box['dec_max'])
            
            # Handle RA wraparound
            if ra_min > ra_max:
                ra_max += 24
            
            width = ra_max - ra_min
            height = dec_max - dec_min
            
            color = 'orange' if box.get('priority', 1) > 1 else 'lightgray'
            alpha = 0.5 if box.get('priority', 1) > 1 else 0.3
            
            rect = Rectangle(
                (ra_min, dec_min), width, height,
                edgecolor='black',
                facecolor=color,
                alpha=alpha,
                label='_nolegend_'
            )
            plt.gca().add_patch(rect)
        
        # Add legend items
        plt.scatter([], [], c='green', alpha=0.7, s=100, label='Completed (with findings)')
        plt.scatter([], [], c='lightgreen', alpha=0.4, s=100, label='Completed (no findings)')
        plt.scatter([], [], c='red', alpha=0.4, s=100, label='Failed')
        plt.scatter([], [], c='orange', alpha=0.5, s=100, label='Pending (high priority)')
        plt.scatter([], [], c='lightgray', alpha=0.3, s=100, label='Pending')
        plt.scatter([], [], c='yellow', marker='*', s=100, edgecolor='black', label='KBO candidate')
        
        # Set up the plot
        plt.xlim(0, 24)
        plt.xlabel('Right Ascension (hours)')
        plt.ylabel('Declination (degrees)')
        plt.title(f'Ecliptic Survey Progress - {survey_state["survey_name"]}')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Add stats as text
        boxes_total = len(survey_state['boxes_searched']) + len(survey_state['next_boxes'])
        boxes_done = len(survey_state['boxes_searched'])
        progress_text = f"Progress: {boxes_done}/{boxes_total} boxes ({boxes_done/boxes_total*100:.1f}%)"
        findings_text = f"Findings: {len(survey_state['findings'])} potential KBO candidates"
        
        plt.figtext(0.02, 0.02, progress_text + '\n' + findings_text, 
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Save the figure
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated survey visualization: {output_file}")
        return output_file
    
    except ImportError:
        logger.error("Cannot generate visualization: matplotlib not installed")
        return None
    except Exception as e:
        logger.error(f"Error generating survey visualization: {e}")
        return None

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Ecliptic KBO Survey Management Tool")
    
    # Main commands
    parser.add_argument('--create', action='store_true', 
                      help='Create a new survey state file')
    parser.add_argument('--run', action='store_true', 
                      help='Run the next iteration of the survey')
    parser.add_argument('--status', action='store_true', 
                      help='Show survey status')
    parser.add_argument('--update-box', metavar='BOX_ID', 
                      help='Update coordinates.txt for a specific box ID')
    parser.add_argument('--report', action='store_true',
                      help='Generate survey report')
    parser.add_argument('--visualize', action='store_true',
                      help='Generate survey progress visualization')
    
    # Options
    parser.add_argument('--force', action='store_true', 
                      help='Force overwrite of existing files')
    parser.add_argument('--survey-file', default=DEFAULT_SURVEY_FILE, 
                      help=f'Survey state file (default: {DEFAULT_SURVEY_FILE})')
    parser.add_argument('--config-file', default=DEFAULT_CONFIG_FILE, 
                      help=f'Coordinates config file (default: {DEFAULT_CONFIG_FILE})')
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR,
                      help=f'Results directory (default: {DEFAULT_RESULTS_DIR})')
    
    # Pipeline options
    parser.add_argument('--download', action='store_true', 
                      help='Download files after search')
    parser.add_argument('--analyze', action='store_true',
                      help='Perform additional analysis')
    parser.add_argument('--ecliptic-priority', action='store_true',
                      help='Prioritize ecliptic proximity')
    parser.add_argument('--visualize-results', action='store_true',
                      help='Generate visualizations of results')
    
    args = parser.parse_args()
    
    # Check for command
    if args.create:
        create_survey_file(args.survey_file, args.force)
    
    elif args.run:
        # Build pipeline arguments
        pipeline_args = []
        if args.download:
            pipeline_args.append('--download')
        if args.ecliptic_priority:
            pipeline_args.append('--ecliptic-priority')
        if args.analyze:
            pipeline_args.append('--analyze')
        if args.visualize_results:
            pipeline_args.append('--visualize')
        
        run_survey_iteration(args.survey_file, pipeline_args, args.results_dir)
    
    elif args.status:
        survey_state = load_survey_file(args.survey_file)
        if survey_state:
            print_survey_summary(survey_state)
    
    elif args.update_box:
        survey_state = load_survey_file(args.survey_file)
        if not survey_state:
            return
        
        # Find the specified box
        target_box = None
        for box in survey_state['next_boxes']:
            if box['box_id'] == args.update_box:
                target_box = box
                break
        
        if not target_box:
            logger.error(f"Box ID not found: {args.update_box}")
            return
        
        update_coordinates_file(target_box, args.config_file)
    
    elif args.report:
        survey_state = load_survey_file(args.survey_file)
        if survey_state:
            report_file = generate_survey_report(survey_state)
            print(f"Report generated: {report_file}")
    
    elif args.visualize:
        survey_state = load_survey_file(args.survey_file)
        if survey_state:
            viz_file = visualize_survey_progress(survey_state)
            if viz_file:
                print(f"Visualization generated: {viz_file}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()