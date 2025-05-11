"""
KBO Catalog Lookup System

This package provides tools to cross-reference KBO candidates against
astronomical catalogs to identify known objects and potential new discoveries.

Main functionality:
- Lookup candidates in multiple astronomical catalogs
- Score matches based on position and motion
- Generate reports for both known objects and potential discoveries
"""

# Version info
__version__ = '0.1.0'
__author__ = 'KBO Hunter Team'

# Core functionality
from catalog_lookup.core.query_manager import lookup_candidate, lookup_candidates
from catalog_lookup.reports.json_reporter import generate_json_report
from catalog_lookup.reports.html_reporter import generate_html_report

# Simplified API for common usage
def check_candidates(candidates_file, output_dir=None, generate_html=True):
    """
    Check KBO candidates against astronomical catalogs and generate reports
    
    Parameters:
    -----------
    candidates_file : str
        Path to JSON file containing KBO candidates
    output_dir : str or None
        Directory to save reports (default: same as candidates_file)
    generate_html : bool
        Whether to generate HTML visual reports
    
    Returns:
    --------
    str
        Path to JSON report file
    """
    # This is a placeholder that will be implemented when we build the modules
    pass

__all__ = [
    'lookup_candidate',
    'lookup_candidates',
    'generate_json_report',
    'generate_html_report',
    'check_candidates',
]