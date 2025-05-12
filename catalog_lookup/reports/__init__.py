"""
Report Generation for Catalog Lookup Results

This module handles the creation of various report formats from
catalog lookup results, including machine-readable JSON, visual HTML,
and MPC submission formats.

Main components:
- JSONReporter: Creates detailed JSON reports with match information
- HTMLReporter: Generates visual HTML reports with images and diagrams
- MPCReporter: Formats potential discoveries for MPC submission
"""

# Import only what currently exists in the reporter modules
# Remove functions that haven't been implemented yet
from catalog_lookup.reports.json_reporter import (
    JSONReporter
    # generate_json_report,  # This function doesn't exist yet
    # parse_json_report      # This function doesn't exist yet
)

from catalog_lookup.reports.html_reporter import (
    HTMLReporter
    # generate_html_report,            # This function doesn't exist yet
    # create_candidate_visualization   # This function doesn't exist yet
)

from catalog_lookup.reports.mpc_reporter import (
    MPCReporter
    # generate_mpc_report,     # This function doesn't exist yet
    # format_mpc_astrometry    # This function doesn't exist yet
)

# Report generation options
DEFAULT_REPORT_OPTIONS = {
    'include_all_matches': True,     # Include all matches, not just the best
    'include_thumbnails': True,      # Include image thumbnails in HTML reports
    'include_orbit_diagrams': True,  # Include orbit diagrams for known objects
    'include_discovery_info': True,  # Include discovery recommendations
    'min_match_score': 0.4           # Minimum score to include a match
}

# Report file naming templates
REPORT_TEMPLATES = {
    'json': '{field_id}_catalog_matches.json',
    'html': '{field_id}_catalog_report.html',
    'mpc': '{field_id}_mpc_submission.txt'
}

# Object classification labels
OBJECT_CLASSIFICATIONS = {
    'classical_kbo': 'Classical Kuiper Belt Object',
    'resonant': 'Resonant KBO',
    'scattered_disk': 'Scattered Disk Object',
    'detached': 'Detached Object',
    'centaur': 'Centaur',
    'plutino': 'Plutino (3:2 resonance)',
    'twotino': 'Twotino (2:1 resonance)',
    'other_resonant': 'Other Resonant KBO',
    'unknown': 'Unclassified Trans-Neptunian Object'
}

__all__ = [
    'JSONReporter',
    'HTMLReporter',
    'MPCReporter',
    'DEFAULT_REPORT_OPTIONS',
    'REPORT_TEMPLATES',
    'OBJECT_CLASSIFICATIONS'
]