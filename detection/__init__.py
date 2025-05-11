"""
Detection package for KBO hunting in JWST data

This package contains modules for detecting Kuiper Belt Objects (KBOs)
using shift-and-stack and other detection algorithms.
"""

# Make modules available at package level
from detection.motion_calculator import calculate_kbo_motion_range, generate_motion_vectors, filter_motion_vectors
from detection.shift_stack import apply_shift, stack_images, process_motion_vector
from detection.candidate_filter import filter_candidates, score_candidate, calculate_kbo_properties
from detection.visualization import visualize_candidates, create_diagnostic_plots

__all__ = [
    'calculate_kbo_motion_range',
    'generate_motion_vectors',
    'filter_motion_vectors',
    'apply_shift',
    'stack_images',
    'process_motion_vector',
    'filter_candidates',
    'score_candidate',
    'calculate_kbo_properties',
    'visualize_candidates',
    'create_diagnostic_plots'
]