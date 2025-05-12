#!/usr/bin/env python3
"""
ecliptic_survey.py - Systematic KBO survey along the ecliptic

This utility manages a systematic survey of the ecliptic plane for KBO detection,
generating search boxes, tracking progress, and coordinating with kbo_hunt.py.
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from datetime import datetime
import numpy as np
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
import astropy.units as u

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ecliptic_survey.log")
    ]
)
logger = logging.getLogger('ecliptic_survey')

# Constants
DEFAULT_SURVEY_FILE = "ecliptic_survey_state.json"
DEFAULT_CONFIG_FILE = "config/coordinates.txt"
DEFAULT_OVERLAP = 0.25  # 25% overlap between boxes

# Priority regions along the ecliptic (RA hours)
# These are regions of particular interest
PRIORITY_ZONES = [
    (2.0, 4.0),    # Opposition in May 2025
    (12.0, 14.0),  # Away from galactic plane
    (18.5, 20.5)   # Known KBO-rich region
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
        'findings': []
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

def run_kbo_hunt(box, pipeline_args=None):
    """
    Run the KBO hunt pipeline on a search box
    
    Parameters:
    -----------
    box : dict
        Search box dictionary
    pipeline_args : list or None
        Additional arguments to pass to kbo_hunt.py
        
    Returns:
    --------
    dict : Results from the pipeline
    """
    try:
        # Construct the command
        cmd = ["python", "kbo_hunt.py", "pipeline", "--config", DEFAULT_CONFIG_FILE]
        
        # Add any additional arguments
        if pipeline_args:
            cmd.extend(pipeline_args)
        
        # Run the command
        logger.info(f"Running KBO hunt pipeline for box {box['box_id']}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Capture the output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Pipeline failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return {
                'success': False,
                'error': result.stderr,
                'output': result.stdout
            }
        
        logger.info(f"Pipeline completed successfully")
        
        # Parse output to find result file paths
        output_lines = result.stdout.split('\n')
        result_files = []
        
        for line in output_lines:
            if "combined_results_" in line and ".json" in line:
                # Extract the path using simple string parsing
                parts = line.split("combined_results_")
                if len(parts) > 1:
                    file_part = parts[1].split(".json")[0] + ".json"
                    result_files.append(os.path.join("data", "search_", file_part))
        
        return {
            'success': True,
            'output': result.stdout,
            'result_files': result_files
        }
        
    except Exception as e:
        logger.error(f"Error running KBO hunt pipeline: {e}")
        return {
            'success': False,
            'error': str(e)
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
    
    # Load the results file if available
    result_files = pipeline_result.get('result_files', [])
    
    if not result_files:
        box['status'] = 'completed_no_results'
        return box
    
    try:
        # Load the most recent result file
        result_file = result_files[-1]
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Extract the summary
        summary = results.get('summary', {})
        
        # Update the box with results
        box['status'] = 'completed'
        box['search_date'] = datetime.now().strftime('%Y-%m-%d')
        box['observations_found'] = summary.get('total_observations', 0)
        box['kbo_candidates'] = summary.get('total_candidates', 0)
        box['sequences_found'] = summary.get('total_sequences', 0)
        box['result_file'] = result_file
        
        # Extract sequences
        sequences = summary.get('sequence_details', [])
        
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
                    'discovery_date': datetime.now().strftime('%Y-%m-%d')
                }
                findings.append(finding)
            
            box['findings'] = findings
        else:
            box['has_findings'] = False
        
        return box
        
    except Exception as e:
        logger.error(f"Error processing pipeline results: {e}")
        box['status'] = 'completed_error'
        box['error'] = str(e)
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

def run_survey_iteration(survey_file=DEFAULT_SURVEY_FILE, pipeline_args=None):
    """
    Run a single iteration of the survey
    
    Parameters:
    -----------
    survey_file : str
        Path to the survey file
    pipeline_args : list or None
        Additional arguments to pass to kbo_hunt.py
        
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
    
    # Run the KBO hunt pipeline
    pipeline_result = run_kbo_hunt(next_box, pipeline_args)
    
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
    """Print a summary of the survey progress"""
    boxes_total = len(survey_state['boxes_searched']) + len(survey_state['next_boxes'])
    boxes_done = len(survey_state['boxes_searched'])
    findings = len(survey_state['findings'])
    
    print("\n=== KBO Ecliptic Survey Summary ===")
    print(f"Survey: {survey_state['survey_name']}")
    print(f"Started: {survey_state['start_date']}")
    print(f"Progress: {boxes_done}/{boxes_total} boxes searched ({boxes_done/boxes_total*100:.1f}%)")
    print(f"Findings: {findings} potential KBO candidates")
    
    if survey_state['boxes_searched']:
        print("\nRecently searched boxes:")
        for box in sorted(survey_state['boxes_searched'][-5:], 
                         key=lambda x: x.get('search_date', ''), reverse=True):
            status = box.get('status', 'unknown')
            obs = box.get('observations_found', 0)
            candidates = box.get('kbo_candidates', 0)
            date = box.get('search_date', 'unknown')
            
            print(f"  {box['box_id']}: {status}, {obs} obs, {candidates} candidates ({date})")
    
    if survey_state['findings']:
        print("\nPotential KBO candidates:")
        for finding in survey_state['findings'][:5]:  # Show top 5
            box_id = finding.get('box_id', 'unknown')
            num_obs = finding.get('num_observations', 0)
            duration = finding.get('duration_hours', 0)
            date = finding.get('discovery_date', 'unknown')
            
            print(f"  {box_id}: {num_obs} observations over {duration:.1f} hours ({date})")
        
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

def main():
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
    
    # Options
    parser.add_argument('--force', action='store_true', 
                      help='Force overwrite of existing files')
    parser.add_argument('--survey-file', default=DEFAULT_SURVEY_FILE, 
                      help=f'Survey state file (default: {DEFAULT_SURVEY_FILE})')
    parser.add_argument('--config-file', default=DEFAULT_CONFIG_FILE, 
                      help=f'Coordinates config file (default: {DEFAULT_CONFIG_FILE})')
    
    # Pipeline options
    parser.add_argument('--download', action='store_true', 
                      help='Download files after search')
    parser.add_argument('--ecliptic-priority', action='store_true',
                      help='Pass --ecliptic-priority to kbo_hunt.py')
    
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
        
        run_survey_iteration(args.survey_file, pipeline_args)
    
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
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()