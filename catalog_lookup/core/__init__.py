"""
Core Catalog Lookup Functionality

This module contains the core logic for querying multiple catalogs,
evaluating matches, and analyzing orbits of potential matches.

Main components:
- QueryManager: Coordinates queries across multiple catalogs
- MatchEvaluator: Assesses match quality between candidates and catalog objects
- OrbitTools: Performs orbital calculations and matching
"""

# Import core components
from catalog_lookup.core.query_manager import QueryManager, lookup_candidate, lookup_candidates
from catalog_lookup.core.match_evaluator import MatchEvaluator, evaluate_match, calculate_match_score
from catalog_lookup.core.orbit_tools import OrbitTools, calculate_orbital_elements, compare_orbits

# Constants used throughout the package
MATCH_THRESHOLDS = {
    'high': 0.85,    # High confidence match
    'medium': 0.65,  # Medium confidence match
    'low': 0.4       # Low confidence match (possible match)
}

# Position match parameters (in arcseconds)
POSITION_TOLERANCES = {
    'tight': 5.0,    # Tight position matching (high precision astrometry)
    'standard': 10.0, # Standard position matching
    'loose': 30.0    # Loose position matching (useful for poorly determined orbits)
}

# Motion match parameters (in arcsec/hour)
MOTION_TOLERANCES = {
    'tight': 0.5,    # Tight motion rate matching
    'standard': 1.0,  # Standard motion rate matching
    'loose': 3.0     # Loose motion rate matching
}

__all__ = [
    'QueryManager',
    'lookup_candidate',
    'lookup_candidates',
    'MatchEvaluator',
    'evaluate_match',
    'calculate_match_score',
    'OrbitTools',
    'calculate_orbital_elements',
    'compare_orbits',
    'MATCH_THRESHOLDS',
    'POSITION_TOLERANCES',
    'MOTION_TOLERANCES'
]