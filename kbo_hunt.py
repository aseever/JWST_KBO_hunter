#!/usr/bin/env python3
"""
kbo_hunt.py - JWST KBO Detection Pipeline

This tool provides an integrated pipeline for searching, filtering, downloading,
and preprocessing JWST FITS data to detect Kuiper Belt Objects.

Example usage:
    # Run the full pipeline with default settings
    python kbo_hunt.py pipeline --config config/coordinates.txt
    
    # Run individual steps
    python kbo_hunt.py search --config config/coordinates.txt --ecliptic-priority
    python kbo_hunt.py filter --catalog data/search_20250511_123456/combined_results_20250511_123456.json
    python kbo_hunt.py download --catalog data/kbo_candidates_20250511_123456.json
    
    # Run preprocessing and detection (uses preprocess.py and kbo_detector.py)
    python kbo_hunt.py preprocess --fits-dir ./data/fits
    python kbo_hunt.py detect --preprocessed-dir ./data/preprocessed
"""

import os
import sys
import argparse
import logging
import json
import subprocess
from datetime import datetime
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kbo_hunt.log")
    ]
)
logger = logging.getLogger('kbo_hunt')

# Constants
DEFAULT_CONFIG_FILE = "config/coordinates.txt"
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_SEARCH_DIR = "search_results"
DEFAULT_FILTER_DIR = "filtered_results"
DEFAULT_DOWNLOAD_DIR = "fits"
DEFAULT_PREPROCESSED_DIR = "preprocessed"
DEFAULT_DETECTION_DIR = "detections"
DEFAULT_VISUALIZATION_DIR = "visualizations"
DEFAULT_MAX_ECLIPTIC_LATITUDE = 5.0

class KBOPipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass

def setup_directories(base_dir=DEFAULT_OUTPUT_DIR):
    """
    Set up the directory structure for pipeline outputs
    
    Parameters:
    -----------
    base_dir : str
        Base directory for all outputs
        
    Returns:
    --------
    dict : Dictionary of directory paths
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base directories with timestamp to avoid overwriting
    search_dir = os.path.join(base_dir, f"{DEFAULT_SEARCH_DIR}_{timestamp}")
    filter_dir = os.path.join(base_dir, f"{DEFAULT_FILTER_DIR}_{timestamp}")
    download_dir = os.path.join(base_dir, DEFAULT_DOWNLOAD_DIR)
    preprocessed_dir = os.path.join(base_dir, DEFAULT_PREPROCESSED_DIR)
    detection_dir = os.path.join(base_dir, f"{DEFAULT_DETECTION_DIR}_{timestamp}")
    viz_dir = os.path.join(base_dir, DEFAULT_VISUALIZATION_DIR)
    
    # Create directories if they don't exist
    for directory in [search_dir, filter_dir, download_dir, preprocessed_dir, detection_dir, viz_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return {
        'base': base_dir,
        'search': search_dir,
        'filter': filter_dir,
        'download': download_dir,
        'preprocessed': preprocessed_dir,
        'detection': detection_dir,
        'visualization': viz_dir,
        'timestamp': timestamp
    }

def handle_search_command(args):
    """
    Handle the 'search' subcommand - search MAST for observations
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Returns:
    --------
    int : Exit code (0 for success)
    """
    # Import required modules
    try:
        from mast.utils import read_coordinates
        from mast.search import search_multiple_squares, prioritize_squares
        from mast.utils import divide_region_into_squares
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure mast package is installed correctly")
        return 1
    
    # Set up output directory
    dirs = setup_directories(args.output_dir)
    search_dir = dirs['search']
    
    logger.info(f"Starting MAST search")
    logger.info(f"Using coordinate configuration: {args.config}")
    logger.info(f"Output directory: {search_dir}")
    
    try:
        # Read coordinates
        coords = read_coordinates(args.config)
        logger.info(f"Searching region: RA {coords['ra_min']:.2f}° to {coords['ra_max']:.2f}°, "
                    f"Dec {coords['dec_min']:.2f}° to {coords['dec_max']:.2f}°")
        
        # Divide region into search squares
        squares = divide_region_into_squares(coords)
        logger.info(f"Divided region into {len(squares)} search squares")
        
        # Prioritize squares if requested
        if args.ecliptic_priority:
            squares = prioritize_squares(squares, ecliptic_priority=True)
            logger.info("Squares prioritized by ecliptic proximity")
        
        # Limit search squares if specified
        if args.max_squares and args.max_squares < len(squares):
            squares = squares[:args.max_squares]
            logger.info(f"Limited search to {args.max_squares} squares")
        
        # Perform search
        results = search_multiple_squares(
            squares,
            output_dir=search_dir,
            max_concurrent=args.parallel,
            timeout=args.timeout
        )
        
        # Create summary file
        summary_file = os.path.join(search_dir, f"search_summary_{dirs['timestamp']}.json")
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Search completed. Results saved to {search_dir}")
        logger.info(f"Summary file: {summary_file}")
        
        # Generate result file path to return
        sequence_file = None
        for filename in os.listdir(search_dir):
            if filename.startswith("sequences_") and filename.endswith(".json"):
                sequence_file = os.path.join(search_dir, filename)
                break
        
        if sequence_file:
            logger.info(f"Generated sequences file: {sequence_file}")
            logger.info(f"To filter these results, run:")
            logger.info(f"  python kbo_hunt.py filter --catalog {sequence_file}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during search: {e}")
        logger.exception("Stack trace:")
        return 1

def handle_filter_command(args):
    """
    Handle the 'filter' subcommand - filter observations for KBO candidates
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Returns:
    --------
    int : Exit code (0 for success)
    """
    # Import required modules
    try:
        from mast.filter.core import filter_catalog
        from mast.filter.visualizations import generate_filter_visualizations
        from mast.filter.analysis import (
            analyze_sequence_coverage,
            estimate_detection_sensitivity,
            analyze_observation_quality
        )
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure mast.filter package is installed correctly")
        return 1
    
    # Set up output directory if not specified
    if not args.output:
        dirs = setup_directories(args.output_dir)
        filter_dir = dirs['filter']
        output_file = os.path.join(filter_dir, f"kbo_candidates_{dirs['timestamp']}.json")
    else:
        output_file = args.output
        filter_dir = os.path.dirname(output_file)
        os.makedirs(filter_dir, exist_ok=True)
    
    viz_dir = os.path.join(filter_dir, "visualizations")
    
    logger.info(f"Starting filtering")
    logger.info(f"Input catalog: {args.catalog}")
    logger.info(f"Output file: {output_file}")
    
    try:
        # Filter the catalog
        filter_results = filter_catalog(
            args.catalog,
            output_file=output_file,
            include_nircam=args.include_nircam,
            max_ecliptic_latitude=args.ecliptic_latitude,
            min_exposure_time=args.min_exposure,
            min_sequence_interval=args.min_interval,
            max_sequence_interval=args.max_interval
        )
        
        if not filter_results:
            logger.error("Filtering failed or produced no results")
            return 1
        
        # Extract sequences
        sequences = filter_results.get('sequences', [])
        stats = filter_results.get('stats', {})
        
        logger.info(f"Filtering summary:")
        logger.info(f"  Initial observations: {stats.get('initial_observations', 0)}")
        logger.info(f"  Filtered observations: {stats.get('filtered_observations', 0)}")
        logger.info(f"  Detected sequences: {stats.get('sequences', len(sequences))}")
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info(f"Generating visualizations")
            viz_files = generate_filter_visualizations(filter_results, viz_dir)
            logger.info(f"Generated {len(viz_files)} visualization files")
        
        # Perform additional analysis if requested
        if args.analyze:
            logger.info(f"Performing additional analysis")
            
            # Get observations from filter results for analysis
            observations = []
            for seq in sequences:
                if 'observations' in seq:
                    observations.extend(seq['observations'])
            
            # Coverage analysis
            coverage = analyze_sequence_coverage(sequences)
            if coverage:
                coverage_file = os.path.join(filter_dir, f"coverage_analysis_{dirs['timestamp']}.json")
                with open(coverage_file, 'w') as f:
                    json.dump(coverage, f, indent=2, default=str)
                logger.info(f"Coverage analysis saved to {coverage_file}")
            
            # Sensitivity analysis
            sensitivity = estimate_detection_sensitivity(sequences)
            if sensitivity:
                sensitivity_file = os.path.join(filter_dir, f"sensitivity_analysis_{dirs['timestamp']}.json")
                with open(sensitivity_file, 'w') as f:
                    json.dump(sensitivity, f, indent=2, default=str)
                logger.info(f"Sensitivity analysis saved to {sensitivity_file}")
                logger.info(f"Overall sensitivity rating: {sensitivity.get('overall_rating', 'Unknown')}")
            
            # Quality analysis
            quality = analyze_observation_quality(observations)
            if quality:
                quality_file = os.path.join(filter_dir, f"quality_analysis_{dirs['timestamp']}.json")
                with open(quality_file, 'w') as f:
                    json.dump(quality, f, indent=2, default=str)
                logger.info(f"Quality analysis saved to {quality_file}")
                
                if 'quality_metrics' in quality and 'quality_rating' in quality['quality_metrics']:
                    logger.info(f"Observation quality rating: {quality['quality_metrics']['quality_rating']}")
        
        # Show download command
        if sequences:
            logger.info(f"\nTo download these KBO candidates, run:")
            logger.info(f"  python kbo_hunt.py download --catalog {output_file}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during filtering: {e}")
        logger.exception("Stack trace:")
        return 1

def handle_download_command(args):
    """
    Handle the 'download' subcommand - download FITS files from MAST
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Returns:
    --------
    int : Exit code (0 for success)
    """
    # Import required modules
    try:
        from mast.download import download_from_catalog, retry_failed_downloads
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure mast package is installed correctly")
        return 1
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        dirs = setup_directories()
        output_dir = dirs['base']
    
    logger.info(f"Starting download")
    logger.info(f"Input catalog: {args.catalog}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Perform download
        status = download_from_catalog(
            args.catalog,
            output_dir=output_dir,
            resume=args.resume,
            delay=args.delay
        )
        
        if not status:
            logger.error("Download failed or status tracking error occurred")
            return 1
        
        # Show summary
        logger.info(f"\nDownload summary:")
        logger.info(f"  Total files: {status.get('total', 0)}")
        logger.info(f"  Downloaded: {status.get('downloaded', 0)}")
        logger.info(f"  Failed: {status.get('failed', 0)}")
        
        # Show preprocess command
        fits_dir = os.path.join(output_dir, "fits")
        if os.path.exists(fits_dir) and os.listdir(fits_dir):
            logger.info(f"\nTo preprocess the downloaded FITS files, run:")
            logger.info(f"  python kbo_hunt.py preprocess --fits-dir {fits_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during download: {e}")
        logger.exception("Stack trace:")
        return 1

def handle_preprocess_command(args):
    """
    Handle the 'preprocess' subcommand - run preprocessing on FITS files
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Returns:
    --------
    int : Exit code (0 for success)
    """
    # Check if preprocess.py exists
    preprocess_script = 'preprocess.py'
    if not os.path.exists(preprocess_script):
        logger.error(f"Preprocessing script not found: {preprocess_script}")
        return 1
    
    # Set up output directory
    if not args.output_dir:
        dirs = setup_directories()
        output_dir = os.path.join(dirs['base'], DEFAULT_PREPROCESSED_DIR)
    else:
        output_dir = args.output_dir
    
    logger.info(f"Starting preprocessing")
    logger.info(f"FITS directory: {args.fits_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Build command
    cmd = [sys.executable, preprocess_script, '--fits-dir', args.fits_dir, '--output-dir', output_dir]
    
    if args.verbose:
        cmd.append('--verbose')
    
    if args.align:
        cmd.append('--align')
        cmd.append(args.align)
    
    if args.background:
        cmd.append('--background')
    
    if args.normalize:
        cmd.append('--normalize')
    
    try:
        # Run preprocessing
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Preprocessing failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return 1
        
        # Log output
        for line in result.stdout.splitlines():
            logger.info(f"Preprocess: {line}")
        
        logger.info(f"Preprocessing completed successfully")
        
        # Show detect command
        logger.info(f"\nTo detect KBOs in the preprocessed data, run:")
        logger.info(f"  python kbo_hunt.py detect --preprocessed-dir {output_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        logger.exception("Stack trace:")
        return 1

def handle_detect_command(args):
    """
    Handle the 'detect' subcommand - run KBO detection on preprocessed data
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Returns:
    --------
    int : Exit code (0 for success)
    """
    # Check if kbo_detector.py exists
    detector_script = 'kbo_detector.py'
    if not os.path.exists(detector_script):
        logger.error(f"Detection script not found: {detector_script}")
        return 1
    
    # Set up output directory
    if not args.output_dir:
        dirs = setup_directories()
        output_dir = os.path.join(dirs['base'], f"{DEFAULT_DETECTION_DIR}_{dirs['timestamp']}")
    else:
        output_dir = args.output_dir
    
    logger.info(f"Starting KBO detection")
    logger.info(f"Preprocessed directory: {args.preprocessed_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Build command
    cmd = [sys.executable, detector_script, '--preprocessed-dir', args.preprocessed_dir, '--output-dir', output_dir]
    
    if args.verbose:
        cmd.append('--verbose')
    
    if args.threshold:
        cmd.append('--threshold')
        cmd.append(str(args.threshold))
    
    if args.min_shift:
        cmd.append('--min-shift')
        cmd.append(str(args.min_shift))
    
    if args.max_shift:
        cmd.append('--max-shift')
        cmd.append(str(args.max_shift))
    
    try:
        # Run detection
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Detection failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return 1
        
        # Log output
        for line in result.stdout.splitlines():
            logger.info(f"Detect: {line}")
        
        logger.info(f"KBO detection completed successfully")
        
        # Show lookup command 
        logger.info(f"\nTo lookup KBO candidates, run:")
        logger.info(f"  python lookup_kbo_candidates.py --candidates-dir {output_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        logger.exception("Stack trace:")
        return 1

def handle_lookup_command(args):
    """
    Handle the 'lookup' subcommand - check for catalog entries of anomalies
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Returns:
    --------
    int : Exit code (0 for success)
    """
    # Check if lookup_kbo_candidates.py exists
    lookup_script = 'lookup_kbo_candidates.py'
    if not os.path.exists(lookup_script):
        logger.error(f"Lookup script not found: {lookup_script}")
        return 1
    
    logger.info(f"Starting KBO candidate lookup")
    logger.info(f"Candidates directory: {args.candidates_dir}")
    
    # Build command
    cmd = [sys.executable, lookup_script, '--candidates-dir', args.candidates_dir]
    
    if args.verbose:
        cmd.append('--verbose')
    
    try:
        # Run lookup
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Lookup failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return 1
        
        # Log output
        for line in result.stdout.splitlines():
            logger.info(f"Lookup: {line}")
        
        logger.info(f"KBO candidate lookup completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Error during lookup: {e}")
        logger.exception("Stack trace:")
        return 1

def handle_pipeline_command(args):
    """
    Handle the 'pipeline' subcommand - run the entire KBO detection pipeline
    
    Parameters:
    -----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Returns:
    --------
    int : Exit code (0 for success)
    """
    # Set up unified directory structure
    dirs = setup_directories(args.output_dir)
    
    logger.info(f"Starting KBO detection pipeline")
    logger.info(f"Using coordinate configuration: {args.config}")
    logger.info(f"Base output directory: {dirs['base']}")
    
    # Step 1: Search
    logger.info("\n=== STEP 1: SEARCH ===")
    search_args = argparse.Namespace(
        config=args.config,
        output_dir=dirs['base'],
        ecliptic_priority=args.ecliptic_priority,
        max_squares=args.max_squares,
        parallel=args.parallel,
        timeout=args.timeout
    )
    
    search_result = handle_search_command(search_args)
    if search_result != 0:
        logger.error("Search step failed, aborting pipeline")
        return search_result
    
    # Find the generated sequence file
    sequence_file = None
    for filename in os.listdir(dirs['search']):
        if filename.startswith("sequences_") and filename.endswith(".json"):
            sequence_file = os.path.join(dirs['search'], filename)
            break
    
    if not sequence_file:
        logger.error("No sequence file found after search, aborting pipeline")
        return 1
    
    # Step 2: Filter
    logger.info("\n=== STEP 2: FILTER ===")
    filter_output = os.path.join(dirs['filter'], f"kbo_candidates_{dirs['timestamp']}.json")
    
    filter_args = argparse.Namespace(
        catalog=sequence_file,
        output=filter_output,
        output_dir=dirs['base'],
        include_nircam=args.include_nircam,
        ecliptic_latitude=args.ecliptic_latitude,
        min_exposure=args.min_exposure,
        min_interval=args.min_interval,
        max_interval=args.max_interval,
        visualize=args.visualize,
        analyze=args.analyze
    )
    
    filter_result = handle_filter_command(filter_args)
    if filter_result != 0:
        logger.error("Filter step failed, aborting pipeline")
        return filter_result
    
    # Step 3: Download (if requested)
    if args.download:
        logger.info("\n=== STEP 3: DOWNLOAD ===")
        download_args = argparse.Namespace(
            catalog=filter_output,
            output_dir=dirs['base'],
            resume=False,
            delay=args.delay
        )
        
        download_result = handle_download_command(download_args)
        if download_result != 0:
            logger.error("Download step failed, aborting pipeline")
            return download_result
        
        # Step 4: Preprocess (if download succeeded)
        logger.info("\n=== STEP 4: PREPROCESS ===")
        fits_dir = os.path.join(dirs['base'], "fits")
        
        preprocess_args = argparse.Namespace(
            fits_dir=fits_dir,
            output_dir=dirs['preprocessed'],
            verbose=args.verbose,
            align=args.align,
            background=args.background,
            normalize=args.normalize
        )
        
        preprocess_result = handle_preprocess_command(preprocess_args)
        if preprocess_result != 0:
            logger.error("Preprocess step failed, aborting pipeline")
            return preprocess_result
        
        # Step 5: Detect (if preprocess succeeded)
        logger.info("\n=== STEP 5: DETECT ===")
        
        detect_args = argparse.Namespace(
            preprocessed_dir=dirs['preprocessed'],
            output_dir=dirs['detection'],
            verbose=args.verbose,
            threshold=args.threshold,
            min_shift=args.min_shift,
            max_shift=args.max_shift
        )
        
        detect_result = handle_detect_command(detect_args)
        if detect_result != 0:
            logger.error("Detect step failed, aborting pipeline")
            return detect_result
        
        # Step 6: Lookup (if detect succeeded)
        logger.info("\n=== STEP 6: LOOKUP ===")
        
        lookup_args = argparse.Namespace(
            candidates_dir=dirs['detection'],
            verbose=args.verbose
        )
        
        lookup_result = handle_lookup_command(lookup_args)
        if lookup_result != 0:
            logger.error("Lookup step failed")
            # Continue anyway as this is the final step
    
    # Pipeline complete
    logger.info("\n=== KBO DETECTION PIPELINE COMPLETED SUCCESSFULLY ===")
    logger.info(f"Results directory: {dirs['base']}")
    
    # Generate summary of results
    summary = {
        'timestamp': dirs['timestamp'],
        'directories': dirs,
        'steps_completed': ['search', 'filter'],
        'config_file': args.config
    }
    
    if args.download:
        summary['steps_completed'].extend(['download', 'preprocess', 'detect', 'lookup'])
    
    summary_file = os.path.join(dirs['base'], f"pipeline_summary_{dirs['timestamp']}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Pipeline summary saved to {summary_file}")
    return 0

def parse_args():
    """
    Parse command line arguments
    
    Returns:
    --------
    argparse.Namespace : Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="JWST KBO Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Pipeline command (runs all steps)
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the full KBO detection pipeline')
    pipeline_parser.add_argument('--config', default=DEFAULT_CONFIG_FILE, help='Path to coordinates configuration file')
    pipeline_parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Base output directory')
    pipeline_parser.add_argument('--download', action='store_true', help='Download and process FITS files')
    pipeline_parser.add_argument('--ecliptic-priority', action='store_true', help='Prioritize squares near ecliptic')
    pipeline_parser.add_argument('--max-squares', type=int, help='Maximum number of squares to search')
    pipeline_parser.add_argument('--include-nircam', action='store_true', help='Include NIRCam observations')
    pipeline_parser.add_argument('--ecliptic-latitude', type=float, default=DEFAULT_MAX_ECLIPTIC_LATITUDE, 
                              help='Maximum ecliptic latitude (degrees)')
    pipeline_parser.add_argument('--min-exposure', type=float, help='Minimum exposure time (seconds)')
    pipeline_parser.add_argument('--min-interval', type=float, help='Minimum sequence interval (hours)')
    pipeline_parser.add_argument('--max-interval', type=float, help='Maximum sequence interval (hours)')
    pipeline_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    pipeline_parser.add_argument('--analyze', action='store_true', help='Perform additional analysis')
    pipeline_parser.add_argument('--delay', type=float, default=1.0, help='Delay between downloads (seconds)')
    pipeline_parser.add_argument('--parallel', type=int, default=1, help='Number of parallel processes')
    pipeline_parser.add_argument('--timeout', type=int, default=300, help='Search timeout (seconds)')
    pipeline_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    pipeline_parser.add_argument('--align', choices=['wcs', 'centroid', 'correlation', 'none'], 
                              help='Alignment method for preprocessing')
    pipeline_parser.add_argument('--background', action='store_true', help='Subtract background in preprocessing')
    pipeline_parser.add_argument('--normalize', action='store_true', help='Normalize images in preprocessing')
    pipeline_parser.add_argument('--threshold', type=float, help='Detection threshold')
    pipeline_parser.add_argument('--min-shift', type=float, help='Minimum shift for detection (pixels)')
    pipeline_parser.add_argument('--max-shift', type=float, help='Maximum shift for detection (pixels)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search MAST for observations')
    search_parser.add_argument('--config', default=DEFAULT_CONFIG_FILE, help='Path to coordinates configuration file')
    search_parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    search_parser.add_argument('--ecliptic-priority', action='store_true', help='Prioritize squares near ecliptic')
    search_parser.add_argument('--max-squares', type=int, help='Maximum number of squares to search')
    search_parser.add_argument('--parallel', type=int, default=1, help='Number of parallel processes')
    search_parser.add_argument('--timeout', type=int, default=300, help='Search timeout (seconds)')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter observations for KBO candidates')
    filter_parser.add_argument('--catalog', required=True, help='Path to catalog file from search')
    filter_parser.add_argument('--output', help='Output file path')
    filter_parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    filter_parser.add_argument('--include-nircam', action='store_true', help='Include NIRCam observations')
    filter_parser.add_argument('--ecliptic-latitude', type=float, default=DEFAULT_MAX_ECLIPTIC_LATITUDE, 
                             help='Maximum ecliptic latitude (degrees)')
    filter_parser.add_argument('--min-exposure', type=float, help='Minimum exposure time (seconds)')
    filter_parser.add_argument('--min-interval', type=float, help='Minimum sequence interval (hours)')
    filter_parser.add_argument('--max-interval', type=float, help='Maximum sequence interval (hours)')
    filter_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    filter_parser.add_argument('--analyze', action='store_true', help='Perform additional analysis')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download FITS files from MAST')
    download_parser.add_argument('--catalog', required=True, help='Path to filtered catalog file')
    download_parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    download_parser.add_argument('--resume', action='store_true', help='Resume previous download')
    download_parser.add_argument('--delay', type=float, default=1.0, help='Delay between downloads (seconds)')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess downloaded FITS files')
    preprocess_parser.add_argument('--fits-dir', required=True, help='Directory containing FITS files')
    preprocess_parser.add_argument('--output-dir', help='Output directory for preprocessed files')
    preprocess_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    preprocess_parser.add_argument('--align', choices=['wcs', 'centroid', 'correlation', 'none'], 
                                 help='Alignment method')
    preprocess_parser.add_argument('--background', action='store_true', help='Subtract background')
    preprocess_parser.add_argument('--normalize', action='store_true', help='Normalize images')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect KBOs in preprocessed data')
    detect_parser.add_argument('--preprocessed-dir', required=True, help='Directory with preprocessed files')
    detect_parser.add_argument('--output-dir', help='Output directory for detection results')
    detect_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    detect_parser.add_argument('--threshold', type=float, help='Detection threshold')
    detect_parser.add_argument('--min-shift', type=float, help='Minimum shift for detection (pixels)')
    detect_parser.add_argument('--max-shift', type=float, help='Maximum shift for detection (pixels)')
    
    # Lookup command
    lookup_parser = subparsers.add_parser('lookup', help='Look up KBO candidates in catalogs')
    lookup_parser.add_argument('--candidates-dir', required=True, help='Directory with candidate detections')
    lookup_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command specified
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    return args

def main():
    """Main entry point for the script"""
    # Parse command line arguments
    args = parse_args()
    
    # Handle the specified command
    try:
        if args.command == 'search':
            return handle_search_command(args)
        elif args.command == 'filter':
            return handle_filter_command(args)
        elif args.command == 'download':
            return handle_download_command(args)
        elif args.command == 'preprocess':
            return handle_preprocess_command(args)
        elif args.command == 'detect':
            return handle_detect_command(args)
        elif args.command == 'lookup':
            return handle_lookup_command(args)
        elif args.command == 'pipeline':
            return handle_pipeline_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    
    except KBOPipelineError as e:
        logger.error(f"Pipeline error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.critical("\n\nOperation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Stack trace:")
        return 1

if __name__ == "__main__":
    sys.exit(main())