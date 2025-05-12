#!/usr/bin/env python3
"""
lookup_kbo_candidates.py - Process KBO candidates from kbo_detector output

This script processes the output from kbo_detector, looking up each candidate
in astronomical catalogs to determine if it's a known object or a potential
new discovery.

The script:
1. Traverses the ./data/detections/ directory structure
2. Processes each moving_object_candidates.json file
3. Looks up each candidate in multiple astronomical catalogs
4. Evaluates matches and classifies candidates
5. Generates reports in a parallel directory structure under ./data/reports/

Usage:
    python lookup_kbo_candidates.py [--options]
"""

import os
import sys
import json
import glob
import argparse
import logging
from datetime import datetime
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from astropy.time import Time

# Import components from the catalog_lookup package
from catalog_lookup.core.query_manager import QueryManager
from catalog_lookup.core.match_evaluator import MatchEvaluator
from catalog_lookup.core.orbit_tools import OrbitTools, OrbitalElements, ObservedPosition

from catalog_lookup.reports.json_reporter import JSONReporter
from catalog_lookup.reports.html_reporter import HTMLReporter
from catalog_lookup.reports.mpc_reporter import MPCReporter

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("lookup_kbo_candidates")

def find_candidate_files(detections_dir: str) -> List[Tuple[str, str]]:
    """
    Find all moving_object_candidates.json files in the detections directory.
    
    Args:
        detections_dir: Base directory containing detection results.
        
    Returns:
        List of tuples (field_id, file_path) for each candidate file.
    """
    logger.info(f"Searching for candidate files in {detections_dir}")
    
    candidate_files = []
    
    # Search pattern for candidate files
    pattern = os.path.join(detections_dir, "**", "moving_object_candidates.json")
    
    # Find all matching files
    for file_path in glob.glob(pattern, recursive=True):
        # Extract field_id from the path
        # The field_id is the directory name containing the json file
        field_id = os.path.basename(os.path.dirname(file_path))
        candidate_files.append((field_id, file_path))
    
    logger.info(f"Found {len(candidate_files)} candidate files")
    return candidate_files

def load_candidates(file_path: str) -> List[Dict[str, Any]]:
    """
    Load KBO candidates from a moving_object_candidates.json file.
    
    Args:
        file_path: Path to the JSON file containing candidates.
        
    Returns:
        List of candidate dictionaries.
    """
    logger.info(f"Loading candidates from {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract candidates from the kbo_detector output format
        candidates = []
        
        if 'candidates' in data:
            candidates = data['candidates']
        elif 'filtered_candidates' in data and isinstance(data.get('filtered_candidates'), int):
            # If 'filtered_candidates' is a count, look for the actual list
            if isinstance(data.get('candidates'), list):
                candidates = data['candidates']
        
        logger.info(f"Loaded {len(candidates)} candidates")
        return candidates
    
    except Exception as e:
        logger.error(f"Error loading candidates from {file_path}: {e}")
        return []

def process_candidate(candidate: Dict[str, Any], 
                    field_id: str,
                    query_manager: QueryManager,
                    match_evaluator: MatchEvaluator,
                    orbit_tools: Optional[OrbitTools] = None) -> Dict[str, Any]:
    """
    Process a single KBO candidate by looking it up in catalogs and evaluating matches.
    
    Args:
        candidate: Dictionary with candidate data.
        field_id: ID of the field where the candidate was detected.
        query_manager: Initialized QueryManager for catalog queries.
        match_evaluator: Initialized MatchEvaluator for match assessment.
        orbit_tools: Optional OrbitTools for orbit calculations.
        
    Returns:
        Dictionary with processed candidate including results.
    """
    # Generate a candidate ID if not present
    if 'objID' in candidate:
        candidate_id = candidate['objID']
    elif 'source_idx' in candidate:
        candidate_id = f"{field_id}_candidate_{candidate['source_idx']}"
    else:
        candidate_id = f"{field_id}_candidate_{id(candidate)}"
    
    logger.info(f"Processing candidate: {candidate_id}")
    
    # Extract position data - kbo_detector outputs use 'xcentroid' and 'ycentroid'
    # We need to convert these to RA and Dec if not already present
    if 'ra' not in candidate or 'dec' not in candidate:
        if 'xcentroid' in candidate and 'ycentroid' in candidate:
            # For now, we'll use the centroid as RA/Dec
            # In a real system, this would use proper WCS transformation
            logger.warning(f"Using centroid coordinates as RA/Dec for {candidate_id}")
            ra = float(candidate['xcentroid'])
            dec = float(candidate['ycentroid'])
        else:
            logger.warning(f"Candidate {candidate_id} missing position information, skipping")
            return {
                'id': candidate_id,
                'field_id': field_id,
                'error': "Missing position information",
                'processed': False
            }
    else:
        ra = float(candidate['ra'])
        dec = float(candidate['dec'])
    
    # Get epoch if available or use current time
    epoch = None
    if 'epoch' in candidate:
        if isinstance(candidate['epoch'], str):
            epoch = Time(candidate['epoch'])
        elif isinstance(candidate['epoch'], Time):
            epoch = candidate['epoch']
    
    if epoch is None:
        # Default to current time if not specified
        epoch = Time.now()
        logger.warning(f"No epoch specified for candidate {candidate_id}, using current time")
    
    # Extract motion information if available
    motion_rate = None
    motion_angle = None
    
    if 'motion_vector' in candidate:
        # Extract motion vector components
        motion_vector = candidate['motion_vector']
        if isinstance(motion_vector, list) and len(motion_vector) == 2:
            dx, dy = motion_vector
            
            # Calculate rate and angle
            motion_rate = np.sqrt(dx**2 + dy**2) * 3600  # Convert to arcsec
            motion_angle = np.degrees(np.arctan2(dy, dx))
            
            logger.info(f"Candidate {candidate_id} motion: {motion_rate:.2f} arcsec/hour at {motion_angle:.1f}°")
    
    # Configure search radius based on position uncertainty
    search_radius = 0.5  # Default: 0.5 degrees
    
    # Query catalogs
    logger.info(f"Querying catalogs for candidate {candidate_id} at RA={ra}, Dec={dec}")
    catalog_results = query_manager.search_by_coordinates(
        ra=ra, 
        dec=dec, 
        radius=search_radius, 
        epoch=epoch, 
        combine_results=True
    )
    
    # Get combined results
    if 'combined' in catalog_results:
        potential_matches = catalog_results['combined'].get('objects', [])
        logger.info(f"Found {len(potential_matches)} potential matches across all catalogs")
    else:
        potential_matches = []
        logger.warning("No combined results available")
    
    # Prepare match input
    catalog_dict = {}
    for catalog in ['mpc', 'jpl', 'skybot', 'panstarrs', 'ossos']:
        if catalog in catalog_results and catalog != 'combined':
            # Extract objects from catalog-specific format
            objects = []
            if catalog == 'mpc' and 'objects' in catalog_results[catalog]:
                objects = catalog_results[catalog]['objects']
            elif catalog == 'jpl' and 'data' in catalog_results[catalog]:
                objects = catalog_results[catalog]['data']
            elif catalog == 'skybot' and 'data' in catalog_results[catalog]:
                objects = catalog_results[catalog]['data']
            elif catalog == 'panstarrs' and 'data' in catalog_results[catalog]:
                objects = catalog_results[catalog]['data']
            elif catalog == 'ossos' and isinstance(catalog_results[catalog], list):
                objects = catalog_results[catalog]
            
            catalog_dict[catalog] = objects
    
    # Create enhanced candidate with additional information
    enhanced_candidate = {
        'id': candidate_id,
        'field_id': field_id,
        'ra': ra,
        'dec': dec,
        'epoch': epoch.iso if isinstance(epoch, Time) else epoch,
        'xcentroid': candidate.get('xcentroid'),
        'ycentroid': candidate.get('ycentroid'),
        'flux': candidate.get('flux'),
        'peak': candidate.get('peak'),
        'score': candidate.get('score'),
        'motion_rate': motion_rate,
        'motion_angle': motion_angle,
        'shifts': candidate.get('shifts')
    }
    
    # Evaluate matches
    logger.info(f"Evaluating matches for candidate {candidate_id}")
    match_result = match_evaluator.evaluate_candidate(enhanced_candidate, catalog_dict)
    
    # Add fit orbit if we have orbit_tools and multiple detections
    fitted_orbit = None
    if orbit_tools and 'shifts' in candidate and len(candidate['shifts']) >= 3:
        try:
            logger.info(f"Fitting orbit for candidate {candidate_id}")
            # Convert shifts to ObservedPosition objects
            positions = []
            
            # Base time (for the first position)
            base_time = epoch if epoch else Time.now()
            
            # Process each shift
            for i, shift in enumerate(candidate['shifts']):
                # Calculate RA/Dec for this position
                # In a real system, this would use proper transformations
                pos_ra = ra + shift[0] / 3600.0  # Approximate conversion
                pos_dec = dec + shift[1] / 3600.0
                
                # Calculate time for this position
                # This is approximate - real system would use actual observation times
                pos_time = base_time + i * 0.1 * u.day  # Assume 0.1 day between observations
                
                positions.append(ObservedPosition(
                    ra=pos_ra,
                    dec=pos_dec,
                    epoch=pos_time,
                    ra_err=0.1,
                    dec_err=0.1
                ))
            
            # Fit orbit if we have enough positions
            if len(positions) >= 3:
                # Default to 40 AU if no distance estimate
                distance_au = 40.0
                
                # Fit the orbit
                fitted_elements, rms_error = orbit_tools.fit_orbit(positions, distance_estimate_au=distance_au)
                
                # Store the fitted orbit
                fitted_orbit = {
                    'elements': {
                        'a': fitted_elements.a,
                        'e': fitted_elements.e,
                        'i': fitted_elements.i,
                        'Omega': fitted_elements.Omega,
                        'omega': fitted_elements.omega,
                        'M': fitted_elements.M,
                        'epoch': fitted_elements.epoch.iso
                    },
                    'derived': {
                        'perihelion': fitted_elements.q,
                        'aphelion': fitted_elements.Q,
                        'period_years': fitted_elements.P
                    },
                    'fit_quality': {
                        'rms_error_arcsec': rms_error,
                        'num_positions': len(positions)
                    },
                    'dynamical_class': orbit_tools.classify_orbit(fitted_elements).name
                }
                
                logger.info(f"Successfully fit orbit for candidate {candidate_id}")
                logger.info(f"Orbital elements: a={fitted_elements.a:.2f} AU, e={fitted_elements.e:.3f}, i={fitted_elements.i:.2f}°")
                logger.info(f"Dynamical class: {fitted_orbit['dynamical_class']}")
                
        except Exception as e:
            logger.error(f"Error fitting orbit for candidate {candidate_id}: {e}")
    
    # Create result dictionary
    result = {
        'id': candidate_id,
        'field_id': field_id,
        'input': enhanced_candidate,
        'original': candidate,
        'match_result': {
            'is_match': match_result.is_match,
            'confidence': match_result.confidence,
            'classification': match_result.classification,
            'match_catalog': match_result.match_catalog,
            'match_object': match_result.match_object,
            'matches': match_result.matches,
            'notes': match_result.notes
        },
        'catalog_results': catalog_results,
        'fitted_orbit': fitted_orbit,
        'processed': True,
        'processing_time': datetime.now().isoformat()
    }
    
    return result

def process_candidate_file(file_path: str, 
                         field_id: str,
                         query_manager: QueryManager,
                         match_evaluator: MatchEvaluator,
                         orbit_tools: Optional[OrbitTools],
                         reports_base_dir: str,
                         formats: List[str]) -> Dict[str, Any]:
    """
    Process a single moving_object_candidates.json file.
    
    Args:
        file_path: Path to the candidates file.
        field_id: ID of the field the candidates are from.
        query_manager: Query manager for catalog searches.
        match_evaluator: Match evaluator for assessing matches.
        orbit_tools: Optional orbit tools for fitting orbits.
        reports_base_dir: Base directory for reports.
        formats: List of report formats to generate.
        
    Returns:
        Dictionary with processing results summary.
    """
    logger.info(f"Processing candidate file: {file_path}")
    
    # Load candidates
    candidates = load_candidates(file_path)
    
    if not candidates:
        logger.warning(f"No candidates found in {file_path}")
        return {
            'field_id': field_id,
            'file_path': file_path,
            'candidates_processed': 0,
            'error': "No candidates found"
        }
    
    # Create field-specific report directories
    field_report_dir = os.path.join(reports_base_dir, field_id)
    os.makedirs(field_report_dir, exist_ok=True)
    
    field_json_dir = os.path.join(field_report_dir, "json")
    field_html_dir = os.path.join(field_report_dir, "html")
    field_mpc_dir = os.path.join(field_report_dir, "mpc")
    
    if 'json' in formats:
        os.makedirs(field_json_dir, exist_ok=True)
    if 'html' in formats:
        os.makedirs(field_html_dir, exist_ok=True)
    if 'mpc' in formats:
        os.makedirs(field_mpc_dir, exist_ok=True)
    
    # Create reporters
    json_reporter = HTMLReporter = mpc_reporter = None
    
    if 'json' in formats:
        json_reporter = JSONReporter(output_dir=field_json_dir)
    
    if 'html' in formats:
        html_reporter = HTMLReporter(output_dir=field_html_dir)
    
    if 'mpc' in formats:
        mpc_reporter = MPCReporter(output_dir=field_mpc_dir)
    
    # Process candidates
    results = []
    for i, candidate in enumerate(candidates):
        logger.info(f"Processing candidate {i+1}/{len(candidates)} from field {field_id}")
        
        # Process the candidate
        result = process_candidate(
            candidate,
            field_id,
            query_manager,
            match_evaluator,
            orbit_tools
        )
        
        results.append(result)
        
        # Generate individual candidate reports
        if result.get('processed', False):
            candidate_id = result['id']
            
            # JSON report
            if 'json' in formats and json_reporter:
                json_path = json_reporter.generate_candidate_report(
                    result['input'], 
                    result['match_result'],
                    result.get('catalog_results')
                )
                logger.debug(f"Generated JSON report for candidate {candidate_id}: {json_path}")
            
            # HTML report
            if 'html' in formats and html_reporter:
                html_path = html_reporter.generate_candidate_report(
                    result['input'],
                    result['match_result'],
                    result.get('catalog_results'),
                    None  # No detection images for now
                )
                logger.debug(f"Generated HTML report for candidate {candidate_id}: {html_path}")
    
    # Classify results
    processed_results = [r for r in results if r.get('processed', False)]
    new_objects = [r for r in processed_results if r['match_result']['classification'] == 'possible_new']
    known_objects = [r for r in processed_results if r['match_result']['classification'].startswith('known_')]
    
    # Create session info for this field
    session_info = {
        'field_id': field_id,
        'file_path': file_path,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'candidates_total': len(candidates),
        'candidates_processed': len(processed_results),
        'possible_new_objects': len(new_objects),
        'known_objects': len(known_objects),
        'processing_errors': len(candidates) - len(processed_results)
    }
    
    # Generate session reports
    if 'json' in formats and json_reporter:
        json_session_path = json_reporter.generate_detection_session_report(session_info, processed_results)
        logger.info(f"Generated JSON session report for field {field_id}: {json_session_path}")
    
    if 'html' in formats and html_reporter:
        html_session_path = html_reporter.generate_detection_session_report(session_info, processed_results)
        logger.info(f"Generated HTML session report for field {field_id}: {html_session_path}")
    
    # Generate MPC report for new objects
    if 'mpc' in formats and mpc_reporter and new_objects:
        # Extract positions from new object candidates
        detections = []
        for obj in new_objects:
            detection = {
                'ra': obj['input']['ra'],
                'dec': obj['input']['dec'],
                'epoch': obj['input']['epoch']
            }
            detections.append(detection)
        
        if detections:
            mpc_path = mpc_reporter.generate_observation_report(
                detections,
                provisional_designation=f"{field_id}_{datetime.now().year}",
                observer_info={'mpc_code': 'XXX'}  # Use default code
            )
            logger.info(f"Generated MPC report for field {field_id} with {len(detections)} new objects: {mpc_path}")
    
    # Save complete results
    results_file = os.path.join(field_report_dir, "lookup_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'session_info': session_info,
            'results': results
        }, f, indent=2, default=str)
    
    logger.info(f"Saved complete results for field {field_id} to {results_file}")
    
    # Return summary
    return session_info

def generate_summary_report(all_field_results: List[Dict[str, Any]], reports_base_dir: str) -> str:
    """
    Generate a summary report of all processed fields.
    
    Args:
        all_field_results: List of field processing results.
        reports_base_dir: Base directory for reports.
        
    Returns:
        Path to the summary report.
    """
    logger.info("Generating summary report for all fields")
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_fields': len(all_field_results),
        'total_candidates': sum(field['candidates_total'] for field in all_field_results),
        'total_processed': sum(field['candidates_processed'] for field in all_field_results),
        'total_new_objects': sum(field['possible_new_objects'] for field in all_field_results),
        'total_known_objects': sum(field['known_objects'] for field in all_field_results),
        'total_errors': sum(field['processing_errors'] for field in all_field_results),
        'field_results': all_field_results
    }
    
    # Save summary report
    summary_file = os.path.join(reports_base_dir, "summary_report.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved summary report to {summary_file}")
    
    # Create HTML summary report
    html_file = os.path.join(reports_base_dir, "summary_report.html")
    
    with open(html_file, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KBO Catalog Lookup Summary Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        .header {{ border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .summary-box {{ display: inline-block; background-color: #f9f9f9; border: 1px solid #ddd;
                      border-radius: 4px; padding: 15px; margin: 0 10px 20px 0; text-align: center; }}
        .summary-box .number {{ font-size: 2em; font-weight: bold; }}
        .summary-box .label {{ font-size: 0.9em; color: #777; }}
        .good {{ color: green; }} .warning {{ color: orange; }} .bad {{ color: red; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>KBO Catalog Lookup Summary Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>Overall Summary</h2>
    <div>
        <div class="summary-box">
            <div class="number">{summary['total_fields']}</div>
            <div class="label">Fields Processed</div>
        </div>
        <div class="summary-box">
            <div class="number">{summary['total_candidates']}</div>
            <div class="label">Total Candidates</div>
        </div>
        <div class="summary-box">
            <div class="number class="good">{summary['total_new_objects']}</div>
            <div class="label">Possible New Objects</div>
        </div>
        <div class="summary-box">
            <div class="number">{summary['total_known_objects']}</div>
            <div class="label">Known Objects</div>
        </div>
        <div class="summary-box">
            <div class="number class="warning">{summary['total_errors']}</div>
            <div class="label">Processing Errors</div>
        </div>
    </div>
    
    <h2>Field Results</h2>
    <table>
        <thead>
            <tr>
                <th>Field ID</th>
                <th>Total Candidates</th>
                <th>Processed</th>
                <th>New Objects</th>
                <th>Known Objects</th>
                <th>Errors</th>
            </tr>
        </thead>
        <tbody>
""")

        # Add rows for each field
        for field in all_field_results:
            f.write(f"""
            <tr>
                <td>{field['field_id']}</td>
                <td>{field['candidates_total']}</td>
                <td>{field['candidates_processed']}</td>
                <td class="good">{field['possible_new_objects']}</td>
                <td>{field['known_objects']}</td>
                <td class="{'warning' if field['processing_errors'] > 0 else ''}">{field['processing_errors']}</td>
            </tr>""")

        # Close the HTML file
        f.write("""
        </tbody>
    </table>
    
    <div style="margin-top: 30px; border-top: 1px solid #eee; padding-top: 10px;">
        <p>Generated by KBO Catalog Lookup System</p>
    </div>
</body>
</html>
""")
    
    logger.info(f"Saved HTML summary report to {html_file}")
    
    return summary_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process KBO candidates from kbo_detector output")
    
    parser.add_argument('--detections-dir', default="./data/detections", 
                       help="Directory containing kbo_detector output")
    parser.add_argument('--reports-dir', default="./data/reports", 
                       help="Directory for output reports")
    parser.add_argument('--field-id', help="Process only this field ID")
    parser.add_argument('--cache-dir', default="./data/cache", 
                       help="Directory for query cache")
    parser.add_argument('--ossos-data-dir', default="./data/ossos_data", 
                       help="Directory containing OSSOS catalog data")
    parser.add_argument('--report-formats', default="json,html", 
                       help="Comma-separated list of report formats: json,html,mpc")
    parser.add_argument('--parallel', action='store_true', 
                       help="Use parallel queries")
    parser.add_argument('--fit-orbits', action='store_true', 
                       help="Fit orbits to candidates with multiple detections")
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse report formats
    formats = [fmt.strip().lower() for fmt in args.report_formats.split(',')]
    
    # Create report directory
    os.makedirs(args.reports_dir, exist_ok=True)
    
    # Find candidate files
    candidate_files = find_candidate_files(args.detections_dir)
    
    if not candidate_files:
        logger.error(f"No candidate files found in {args.detections_dir}")
        return 1
    
    # Filter by field_id if specified
    if args.field_id:
        candidate_files = [(field_id, file_path) for field_id, file_path in candidate_files
                          if field_id == args.field_id]
        
        if not candidate_files:
            logger.error(f"No candidate files found for field ID: {args.field_id}")
            return 1
    
    # Initialize core components
    logger.info("Initializing catalog lookup system components...")
    
    # Create query manager
    query_manager = QueryManager(
        cache_dir=args.cache_dir,
        ossos_data_dir=args.ossos_data_dir,
        parallel_queries=args.parallel,
        max_workers=4,
        verbose=args.verbose
    )
    
    # Create match evaluator
    match_evaluator = MatchEvaluator(
        position_tolerance_arcsec=10.0,
        proper_motion_tolerance_percent=20.0,
        verbose=args.verbose
    )
    
    # Create orbit tools if orbit fitting is requested
    orbit_tools = None
    if args.fit_orbits:
        logger.info("Initializing orbit tools for fitting...")
        orbit_tools = OrbitTools(verbose=args.verbose)
    
    # Process each candidate file
    all_field_results = []
    
    for field_id, file_path in candidate_files:
        logger.info(f"Processing field: {field_id}")
        
        field_result = process_candidate_file(
            file_path,
            field_id,
            query_manager,
            match_evaluator,
            orbit_tools,
            args.reports_dir,
            formats
        )
        
        all_field_results.append(field_result)
    
    # Generate summary report
    summary_file = generate_summary_report(all_field_results, args.reports_dir)
    
    # Print summary
    print("\n=== KBO Catalog Lookup Summary ===")
    print(f"Total fields processed: {len(all_field_results)}")
    
    total_candidates = sum(field['candidates_total'] for field in all_field_results)
    total_processed = sum(field['candidates_processed'] for field in all_field_results)
    total_new = sum(field['possible_new_objects'] for field in all_field_results)
    total_known = sum(field['known_objects'] for field in all_field_results)
    total_errors = sum(field['processing_errors'] for field in all_field_results)
    
    print(f"Total candidates: {total_candidates}")
    print(f"Successfully processed: {total_processed}")
    print(f"Possible new objects: {total_new}")
    print(f"Known objects: {total_known}")
    print(f"Processing errors: {total_errors}")
    
    print(f"\nDetailed report available at: {summary_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())