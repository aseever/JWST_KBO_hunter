#!/usr/bin/env python3
"""
kbo_hunt.py - Main control harness for KBO hunting with JWST data

This script provides a command-line interface for searching, filtering,
and downloading JWST observations for KBO detection. It orchestrates the
entire workflow by delegating to specialized modules in the mast package.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import json

# Import mast package modules
try:
    from mast.utils import (
        setup_logging, 
        read_coordinates, 
        format_coordinates, 
        divide_region_into_squares,
        prioritize_squares,
        KBO_DETECTION_CONSTANTS,
        generate_timestamp
    )
    from mast.search import search_multiple_squares
    from mast.filter import filter_catalog, generate_filter_visualizations
    from mast.download import download_from_catalog, retry_failed_downloads
except ImportError as e:
    print(f"Error importing mast package: {e}")
    print("Make sure mast package is in your Python path or in the same directory.")
    sys.exit(1)

def setup_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="JWST KBO Hunt - Search for KBOs in JWST data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for JWST observations in a region:
  python kbo_hunt.py search --config config/coordinates.txt
  
  # Filter observations for KBO candidates:
  python kbo_hunt.py filter --catalog data/combined_results_20250510_123045.json
  
  # Download filtered observations:
  python kbo_hunt.py download --catalog data/kbo_candidates_20250510_124530.json
  
  # Run the entire pipeline in one command:
  python kbo_hunt.py pipeline --config config/coordinates.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search MAST for JWST observations')
    search_parser.add_argument('--config', required=True, help='Configuration file with search coordinates')
    search_parser.add_argument('--output-dir', '-o', default="./data", help='Output directory (default: ./data)')
    search_parser.add_argument('--timeout', type=int, default=300, help='Search timeout in seconds (default: 300)')
    search_parser.add_argument('--max-squares', type=int, help='Maximum number of squares to search (for testing)')
    search_parser.add_argument('--ecliptic-priority', action='store_true', help='Prioritize squares near ecliptic plane')
    search_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter observations for KBO candidates')
    filter_parser.add_argument('--catalog', required=True, help='Path to catalog file from search')
    filter_parser.add_argument('--output-file', '-o', help='Output file (default: auto-generated)')
    filter_parser.add_argument('--include-nircam', action='store_true', help='Include NIRCam observations')
    filter_parser.add_argument('--max-ecliptic', type=float, default=5.0, 
                             help='Maximum ecliptic latitude in degrees (default: 5.0)')
    filter_parser.add_argument('--min-exposure', type=float, help='Minimum exposure time in seconds')
    filter_parser.add_argument('--min-interval', type=float, help='Minimum time between observations in hours')
    filter_parser.add_argument('--max-interval', type=float, help='Maximum time between observations in hours')
    filter_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download FITS files')
    download_parser.add_argument('--catalog', required=True, help='Path to filtered catalog file')
    download_parser.add_argument('--output-dir', '-o', default="./data", help='Output directory (default: ./data)')
    download_parser.add_argument('--resume', action='store_true', help='Resume interrupted download')
    download_parser.add_argument('--retry', action='store_true', help='Retry failed downloads')
    download_parser.add_argument('--delay', type=float, default=1.0, 
                               help='Delay between downloads in seconds (default: 1.0)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check download status')
    status_parser.add_argument('--status-file', required=True, help='Path to download status file')
    
    # Pipeline command (runs all steps)
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline (search, filter, download)')
    pipeline_parser.add_argument('--config', required=True, help='Configuration file with search coordinates')
    pipeline_parser.add_argument('--output-dir', '-o', default="./data", help='Output directory (default: ./data)')
    pipeline_parser.add_argument('--download', action='store_true', help='Download files after filtering')
    pipeline_parser.add_argument('--max-squares', type=int, help='Maximum number of squares to search (for testing)')
    pipeline_parser.add_argument('--ecliptic-priority', action='store_true', help='Prioritize squares near ecliptic plane')
    
    return parser

def command_search(args):
    """Run the search command"""
    logger = logging.getLogger('mast_kbo')
    
    try:
        # Read and validate coordinates
        coords = read_coordinates(args.config)
        formatted_coords = format_coordinates(coords)
        
        logger.info(f"Search box: RA [{formatted_coords['ra_min_hms']} to {formatted_coords['ra_max_hms']}], "
                   f"Dec [{formatted_coords['dec_min_dms']} to {formatted_coords['dec_max_dms']}]")
        
        # Divide region into squares
        squares = divide_region_into_squares(coords)
        
        # Limit number of squares if requested
        if args.max_squares and len(squares) > args.max_squares:
            logger.info(f"Limiting search to {args.max_squares} squares (out of {len(squares)})")
            squares = squares[:args.max_squares]
        
        # Prioritize squares if requested
        if args.ecliptic_priority:
            logger.info("Prioritizing squares near ecliptic plane")
            squares = prioritize_squares(squares, ecliptic_priority=True)
        
        # Create output directory
        timestamp = generate_timestamp()
        output_dir = os.path.join(args.output_dir, f"search_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save search parameters
        params = {
            'timestamp': timestamp,
            'config_file': args.config,
            'coordinates': coords,
            'num_squares': len(squares),
            'timeout': args.timeout,
            'ecliptic_priority': args.ecliptic_priority
        }
        
        params_file = os.path.join(output_dir, "search_params.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"Saved search parameters to {params_file}")
        
        # Run the search
        logger.info(f"Starting search of {len(squares)} squares with {args.timeout}s timeout")
        results = search_multiple_squares(
            squares, 
            output_dir=output_dir,
            timeout=args.timeout
        )
        
        # Save summary
        summary_file = os.path.join(output_dir, "search_summary.json")
        with open(summary_file, 'w') as f:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_observations': results['summary']['total_observations'],
                'total_candidates': results['summary']['total_candidates'],
                'total_sequences': results['summary']['total_sequences'],
                'num_squares': len(squares),
                'squares_with_observations': sum(1 for r in results['results'] if 'total_observations' in r and r['total_observations'] > 0)
            }
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved search summary to {summary_file}")
        
        # Print summary
        logger.info("\n=== Search Summary ===")
        logger.info(f"Total observations found: {results['summary']['total_observations']}")
        logger.info(f"Potential KBO candidates: {results['summary']['total_candidates']}")
        logger.info(f"Observation sequences: {results['summary']['total_sequences']}")
        logger.info(f"Squares with observations: {summary['squares_with_observations']}/{len(squares)}")
        
        # Generate visualizations if requested
        if args.visualize:
            # TODO: Implement visualization generation
            logger.info("Visualization generation not yet implemented")
        
        # Get combined results file path
        results_files = [f for f in os.listdir(output_dir) if f.startswith("combined_results_")]
        if results_files:
            combined_file = os.path.join(output_dir, results_files[0])
            logger.info(f"\nTo filter these results, run:")
            logger.info(f"python kbo_hunt.py filter --catalog {combined_file}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in search command: {e}")
        import traceback
        traceback.print_exc()
        return 1

def command_filter(args):
    """Run the filter command"""
    logger = logging.getLogger('mast_kbo')
    
    try:
        # Run filtering
        logger.info(f"Filtering catalog: {args.catalog}")
        
        results = filter_catalog(
            args.catalog,
            output_file=args.output_file,
            include_nircam=args.include_nircam,
            max_ecliptic_latitude=args.max_ecliptic,
            min_exposure_time=args.min_exposure,
            min_sequence_interval=args.min_interval,
            max_sequence_interval=args.max_interval
        )
        
        if not results:
            logger.error("Filtering failed")
            return 1
        
        # Print summary
        stats = results['stats']
        logger.info("\n=== Filter Summary ===")
        logger.info(f"Initial observations: {stats['initial_observations']}")
        logger.info(f"Filtered observations: {stats['filtered_observations']}")
        logger.info(f"Observation fields: {stats['fields']}")
        logger.info(f"KBO candidate sequences: {stats['sequences']}")
        
        # Generate visualizations if requested
        if args.visualize:
            vis_dir = os.path.join(os.path.dirname(args.output_file or args.catalog), "visualizations")
            vis_files = generate_filter_visualizations(results, output_dir=vis_dir)
            logger.info(f"Generated {len(vis_files)} visualization files in {vis_dir}")
        
        # Get output file path
        if args.output_file:
            output_file = args.output_file
        else:
            # Find the generated output file
            output_dir = os.path.dirname(args.catalog)
            filtered_files = [f for f in os.listdir(output_dir) if f.endswith("_filtered.json")]
            if filtered_files:
                output_file = os.path.join(output_dir, filtered_files[-1])
            else:
                output_file = "unknown_output.json"
        
        logger.info(f"\nTo download these observations, run:")
        logger.info(f"python kbo_hunt.py download --catalog {output_file}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in filter command: {e}")
        import traceback
        traceback.print_exc()
        return 1

def command_download(args):
    """Run the download command"""
    logger = logging.getLogger('mast_kbo')
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.retry:
            # Find status file if retry requested
            status_files = [f for f in os.listdir(os.path.join(args.output_dir, "status")) 
                          if f.startswith("download_status_")]
            
            if not status_files:
                logger.error("No download status files found for retry")
                return 1
            
            # Use the most recent status file
            status_files.sort(reverse=True)
            status_file = os.path.join(args.output_dir, "status", status_files[0])
            
            logger.info(f"Retrying failed downloads from {status_file}")
            status = retry_failed_downloads(status_file, args.output_dir, delay=args.delay)
        else:
            # Start or resume normal download
            logger.info(f"Downloading from catalog: {args.catalog}")
            status = download_from_catalog(
                args.catalog,
                args.output_dir,
                resume=args.resume,
                delay=args.delay
            )
        
        if not status:
            logger.error("Download failed")
            return 1
        
        # Print summary
        logger.info("\n=== Download Summary ===")
        logger.info(f"Total files: {status['total']}")
        logger.info(f"Downloaded: {status['downloaded']}")
        logger.info(f"Failed: {status['failed']}")
        logger.info(f"Remaining: {status['total'] - status['downloaded']}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in download command: {e}")
        import traceback
        traceback.print_exc()
        return 1

def command_status(args):
    """Run the status command"""
    logger = logging.getLogger('mast_kbo')
    
    try:
        # Load status file
        with open(args.status_file, 'r') as f:
            status = json.load(f)
        
        # Print summary
        logger.info("\n=== Download Status ===")
        logger.info(f"Catalog: {status.get('catalog', 'Unknown')}")
        logger.info(f"Started: {status.get('start_time', 'Unknown')}")
        logger.info(f"Last updated: {status.get('last_updated', 'Unknown')}")
        logger.info(f"Total files: {status.get('total', 0)}")
        logger.info(f"Downloaded: {status.get('downloaded', 0)}")
        logger.info(f"Failed: {status.get('failed', 0) or len(status.get('failed_urls', []))}")
        logger.info(f"Remaining: {status.get('total', 0) - status.get('downloaded', 0)}")
        logger.info(f"In progress: {status.get('in_progress', False)}")
        
        # Show failed URLs
        failed_urls = status.get('failed_urls', [])
        if failed_urls:
            logger.info(f"\nFailed downloads ({len(failed_urls)}):")
            for i, url in enumerate(failed_urls[:10]):
                logger.info(f"  {i+1}. {url}")
            
            if len(failed_urls) > 10:
                logger.info(f"  ... and {len(failed_urls)-10} more")
            
            logger.info("\nTo retry failed downloads:")
            logger.info(f"python kbo_hunt.py download --catalog {status.get('catalog', 'catalog.json')} --retry")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in status command: {e}")
        import traceback
        traceback.print_exc()
        return 1

def command_pipeline(args):
    """Run the full pipeline command"""
    logger = logging.getLogger('mast_kbo')
    
    try:
        # Create output directory
        timestamp = generate_timestamp()
        pipeline_dir = os.path.join(args.output_dir, f"pipeline_{timestamp}")
        os.makedirs(pipeline_dir, exist_ok=True)
        
        # Save pipeline parameters
        params = {
            'timestamp': timestamp,
            'config_file': args.config,
            'download': args.download,
            'max_squares': args.max_squares,
            'ecliptic_priority': args.ecliptic_priority
        }
        
        params_file = os.path.join(pipeline_dir, "pipeline_params.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"Starting pipeline with output to {pipeline_dir}")
        
        # Step 1: Search
        logger.info("\n=== Step 1: Search ===")
        
        # Create args object for search command
        search_args = argparse.Namespace(
            config=args.config,
            output_dir=pipeline_dir,
            timeout=300,
            max_squares=args.max_squares,
            ecliptic_priority=args.ecliptic_priority,
            visualize=True
        )
        
        search_result = command_search(search_args)
        if search_result != 0:
            logger.error("Search step failed")
            return 1
        
        # Find the combined results file
        search_dir = [d for d in os.listdir(pipeline_dir) if d.startswith("search_")]
        if not search_dir:
            logger.error("No search results directory found")
            return 1
        
        search_dir = os.path.join(pipeline_dir, search_dir[0])
        combined_files = [f for f in os.listdir(search_dir) if f.startswith("combined_results_")]
        if not combined_files:
            logger.error("No combined results file found")
            return 1
        
        combined_file = os.path.join(search_dir, combined_files[0])
        
        # Step 2: Filter
        logger.info("\n=== Step 2: Filter ===")
        
        filter_output = os.path.join(pipeline_dir, f"kbo_candidates_{timestamp}.json")
        
        # Create args object for filter command
        filter_args = argparse.Namespace(
            catalog=combined_file,
            output_file=filter_output,
            include_nircam=False,
            max_ecliptic=5.0,
            min_exposure=None,
            min_interval=None,
            max_interval=None,
            visualize=True
        )
        
        filter_result = command_filter(filter_args)
        if filter_result != 0:
            logger.error("Filter step failed")
            return 1
        
        # Step 3: Download (optional)
        if args.download:
            logger.info("\n=== Step 3: Download ===")
            
            # Create args object for download command
            download_args = argparse.Namespace(
                catalog=filter_output,
                output_dir=pipeline_dir,
                resume=False,
                retry=False,
                delay=1.0
            )
            
            download_result = command_download(download_args)
            if download_result != 0:
                logger.error("Download step failed")
                return 1
        
        # Pipeline complete
        logger.info("\n=== Pipeline Complete ===")
        logger.info(f"All outputs saved to {pipeline_dir}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in pipeline command: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Main function"""
    # Set up argument parser
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_dir = os.path.join(args.output_dir if hasattr(args, 'output_dir') else 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"kbo_hunt_{args.command}_{timestamp}.log")
    
    logger = setup_logging(log_level=logging.INFO, log_file=log_file)
    
    # Print banner
    logger.info("=" * 60)
    logger.info("JWST KBO Hunt - Kuiper Belt Object Detection Pipeline")
    logger.info("=" * 60)
    logger.info(f"Command: {args.command}")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Execute command
    if args.command == 'search':
        return command_search(args)
    elif args.command == 'filter':
        return command_filter(args)
    elif args.command == 'download':
        return command_download(args)
    elif args.command == 'status':
        return command_status(args)
    elif args.command == 'pipeline':
        return command_pipeline(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())