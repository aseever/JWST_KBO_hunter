"""
mast/filter/__init__.py - KBO filtering package for JWST data

This package provides tools for filtering and analyzing JWST observations
to identify promising Kuiper Belt Object (KBO) candidates. It supports
filtering based on instrument, wavelength, exposure time, and ecliptic
proximity, as well as identifying observation sequences.

This module maintains backward compatibility with the original mast/filter.py
while providing a more modular and extensible architecture.
"""

# Version information
__version__ = '0.2.0'

# Import core functionality
from .core import (
    filter_by_instrument,
    filter_by_wavelength,
    filter_by_exposure,
    filter_by_ecliptic_latitude,
    filter_observations,
    filter_catalog,
    extract_observations_from_catalog,
    quick_filter
)

# Import sequence functionality
from .sequences import (
    group_observations_by_field,
    find_observation_sequences,
    calculate_kbo_motion_parameters,
    score_sequences,
    compute_sequence_statistics,
    find_similar_sequences,
    merge_sequences
)

# Import visualization functionality
from .visualizations import (
    generate_filter_visualizations,
    visualize_sequence_lengths,
    visualize_sequence_durations,
    visualize_sequence_scores,
    visualize_filter_effectiveness,
    create_sequence_dashboard,
    visualize_sequence_time_distribution,
    visualize_spatial_distribution
)

# Import analysis functionality
from .analysis import (
    analyze_sequence_coverage,
    estimate_detection_sensitivity,
    analyze_near_misses,
    analyze_observation_quality,
    evaluate_filter_parameters,
    generate_motion_models,
    plot_motion_model_results
)

# For backward compatibility
# Re-export the main filter_catalog function at module level
from .core import filter_catalog as filter_catalog_from_file

# Define publicly exported names
__all__ = [
    # Core filtering
    'filter_by_instrument',
    'filter_by_wavelength',
    'filter_by_exposure',
    'filter_by_ecliptic_latitude', 
    'filter_observations',
    'filter_catalog',
    'filter_catalog_from_file',
    'extract_observations_from_catalog',
    'quick_filter',
    
    # Sequence handling
    'group_observations_by_field',
    'find_observation_sequences',
    'calculate_kbo_motion_parameters',
    'score_sequences',
    'compute_sequence_statistics',
    'find_similar_sequences',
    'merge_sequences',
    
    # Visualizations
    'generate_filter_visualizations',
    'visualize_sequence_lengths',
    'visualize_sequence_durations',
    'visualize_sequence_scores',
    'visualize_filter_effectiveness',
    'create_sequence_dashboard',
    'visualize_sequence_time_distribution',
    'visualize_spatial_distribution',
    
    # Analysis
    'analyze_sequence_coverage',
    'estimate_detection_sensitivity',
    'analyze_near_misses',
    'analyze_observation_quality',
    'evaluate_filter_parameters',
    'generate_motion_models',
    'plot_motion_model_results'
]

# Backward compatibility for original function arguments
def generate_filter_visualizations_compat(results, output_dir=None):
    """Compatibility wrapper for the original API"""
    return generate_filter_visualizations(results, output_dir)