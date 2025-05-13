"""
mast/filter/stats.py - Statistics tracking for KBO filter pipeline

This module provides classes and functions for tracking statistics throughout
the KBO filtering process, enabling transparency and analysis of filter effectiveness.
"""

import logging
from collections import Counter, defaultdict
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

# Set up logger
logger = logging.getLogger('mast_kbo')

class FilterStats:
    """
    Class to track filtering statistics for transparency
    
    This class maintains statistics throughout the filtering pipeline,
    including counts of observations passing each filter, distributions
    of key parameters, and near misses that just barely fail filters.
    """
    def __init__(self):
        """Initialize a new FilterStats instance"""
        self.initial_count = 0
        self.steps = []  # List of dictionaries with step info
        self.distributions = {}  # Stores distributions of values
        self.near_misses = {}  # Stores near misses by filter type
        self.rejected_reasons = Counter()  # Counts rejection reasons
        
    def add_step(self, name, passed_count, total_count, near_misses=None):
        """
        Record a filtering step with counts and near misses
        
        Parameters:
        -----------
        name : str
            Name of the filtering step
        passed_count : int
            Number of observations that passed this step
        total_count : int
            Total number of observations before this step
        near_misses : dict, list, or int
            Near miss information for this step
        """
        self.steps.append({
            'name': name,
            'passed': passed_count,
            'total': total_count,
            'pass_rate': passed_count / total_count if total_count > 0 else 0,
            'reject_rate': 1 - (passed_count / total_count) if total_count > 0 else 0
        })
        
        if near_misses:
            self.near_misses[name] = near_misses
    
    def add_distribution(self, name, values, threshold=None):
        """
        Record a distribution of values for visualization
        
        Parameters:
        -----------
        name : str
            Name of the distribution parameter
        values : list
            List of values to visualize
        threshold : float, list, or None
            Threshold value(s) for this parameter
        """
        self.distributions[name] = {
            'values': values,
            'threshold': threshold
        }
    
    def add_rejection_reason(self, reason, count=1):
        """
        Track reasons for rejection
        
        Parameters:
        -----------
        reason : str
            Description of the rejection reason
        count : int
            Number of observations rejected for this reason
        """
        self.rejected_reasons[reason] += count
    
    def summary(self):
        """
        Generate a text summary of filtering results
        
        Returns:
        --------
        str : Formatted summary text
        """
        summary_text = "=== Filter Statistics Summary ===\n"
        summary_text += f"Starting with {self.initial_count} observations\n\n"
        
        for step in self.steps:
            summary_text += f"{step['name']}: {step['passed']}/{step['total']} " \
                           f"passed ({step['pass_rate']*100:.1f}%)\n"
        
        summary_text += "\nTop rejection reasons:\n"
        for reason, count in self.rejected_reasons.most_common(5):
            summary_text += f"- {reason}: {count} observations\n"
        
        if self.near_misses:
            summary_text += "\nNear misses by filter:\n"
            for filter_name, count in self.near_misses.items():
                if isinstance(count, (int, float)):
                    summary_text += f"- {filter_name}: {count} observations\n"
                elif isinstance(count, list):
                    summary_text += f"- {filter_name}: {len(count)} observations\n"
                else:
                    summary_text += f"- {filter_name}: present\n"
        
        return summary_text
    
    def get_distribution_stats(self, name):
        """
        Get statistical information about a distribution
        
        Parameters:
        -----------
        name : str
            Name of the distribution
            
        Returns:
        --------
        dict : Dictionary with statistical properties
        """
        if name not in self.distributions:
            return None
            
        values = self.distributions[name]['values']
        
        # Filter out non-numeric values
        if isinstance(values[0], (int, float)):
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if not numeric_values:
                return None
                
            stats = {
                'min': np.min(numeric_values),
                'max': np.max(numeric_values),
                'mean': np.mean(numeric_values),
                'median': np.median(numeric_values),
                'std': np.std(numeric_values),
                'count': len(numeric_values)
            }
            
            # Calculate percentiles
            stats['percentiles'] = {
                '5': np.percentile(numeric_values, 5),
                '25': np.percentile(numeric_values, 25),
                '50': np.percentile(numeric_values, 50),
                '75': np.percentile(numeric_values, 75),
                '95': np.percentile(numeric_values, 95)
            }
            
            return stats
        else:
            # For categorical data, count frequencies
            counter = Counter(values)
            return {
                'frequencies': counter.most_common(),
                'count': len(values),
                'unique_count': len(counter)
            }
    
    def merge(self, other_stats):
        """
        Merge another FilterStats object into this one
        
        Parameters:
        -----------
        other_stats : FilterStats
            Another FilterStats object to merge
            
        Returns:
        --------
        FilterStats : Self, with merged data
        """
        # Update initial count
        if self.initial_count == 0:
            self.initial_count = other_stats.initial_count
        
        # Merge steps - this is tricky, so we'll append them
        self.steps.extend(other_stats.steps)
        
        # Merge distributions - only if they don't already exist
        for name, dist_data in other_stats.distributions.items():
            if name not in self.distributions:
                self.distributions[name] = dist_data
        
        # Merge near misses
        for filter_name, misses in other_stats.near_misses.items():
            if filter_name not in self.near_misses:
                self.near_misses[filter_name] = misses
            elif isinstance(misses, list) and isinstance(self.near_misses[filter_name], list):
                self.near_misses[filter_name].extend(misses)
        
        # Merge rejection reasons
        self.rejected_reasons.update(other_stats.rejected_reasons)
        
        return self


def detect_near_misses(observations, params):
    """
    Detect observations that narrowly miss filter criteria
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    params : dict
        Dictionary of filter parameters with thresholds
        
    Returns:
    --------
    dict : Dictionary of near misses by parameter
    """
    near_misses = defaultdict(list)
    
    # Extract parameters with default values
    max_ecliptic_latitude = params.get('max_ecliptic_latitude', 5.0)
    min_exposure_time = params.get('min_exposure_time', 300)
    min_wavelength = params.get('min_wavelength', 10.0)
    max_wavelength = params.get('max_wavelength', 25.0)
    
    # Margins for near misses
    ecliptic_margin = 2.0  # degrees
    exposure_margin = 0.2  # 20% of threshold
    wavelength_margin = 2.0  # microns
    
    # Find coordinate fields
    ra_field, dec_field = None, None
    coordinate_fields = [('s_ra', 's_dec'), ('ra', 'dec'), ('RA', 'DEC')]
    
    if observations:
        for ra_f, dec_f in coordinate_fields:
            if ra_f in observations[0] and dec_f in observations[0]:
                ra_field, dec_field = ra_f, dec_f
                break
    
    # Find exposure field
    exposure_field = None
    exposure_fields = ['t_exptime', 'exptime', 'exposure_time']
    
    if observations:
        for field in exposure_fields:
            if field in observations[0]:
                exposure_field = field
                break
    
    # Process each observation
    for obs in observations:
        # Check ecliptic latitude if coordinates available
        if ra_field and dec_field:
            try:
                ra = float(obs[ra_field])
                dec = float(obs[dec_field])
                
                # Calculate ecliptic latitude
                coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
                ecliptic_coord = coord.transform_to('ecliptic')
                ecliptic_lat = ecliptic_coord.lat.deg
                
                # Check if it's a near miss
                if max_ecliptic_latitude < abs(ecliptic_lat) <= max_ecliptic_latitude + ecliptic_margin:
                    near_misses['ecliptic_latitude'].append({
                        'id': obs.get('obs_id', 'unknown'),
                        'value': ecliptic_lat,
                        'threshold': max_ecliptic_latitude,
                        'margin': ecliptic_margin
                    })
            except (ValueError, TypeError):
                pass
        
        # Check exposure time if field is available
        if exposure_field:
            try:
                exp_time = float(obs.get(exposure_field, 0))
                
                # Check if it's a near miss
                near_miss_exp = min_exposure_time * (1 - exposure_margin)
                if near_miss_exp <= exp_time < min_exposure_time:
                    near_misses['exposure_time'].append({
                        'id': obs.get('obs_id', 'unknown'),
                        'value': exp_time,
                        'threshold': min_exposure_time,
                        'margin': exposure_margin
                    })
            except (ValueError, TypeError):
                pass
        
        # Check wavelength if available
        if 'wavelength_range' in obs and obs['wavelength_range']:
            try:
                wavelength = obs['wavelength_range']
                if isinstance(wavelength, (list, tuple, np.ndarray)) and len(wavelength) >= 2:
                    wl_min, wl_max = wavelength[0], wavelength[1]
                    
                    # Convert to microns if needed
                    if wl_min < 1e-4 or wl_max < 1e-4:  # Likely in meters
                        wl_min *= 1e6
                        wl_max *= 1e6
                    
                    # Check for near misses
                    if wl_max < min_wavelength and wl_max >= min_wavelength - wavelength_margin:
                        near_misses['wavelength_min'].append({
                            'id': obs.get('obs_id', 'unknown'),
                            'value': wl_max,
                            'threshold': min_wavelength,
                            'margin': wavelength_margin
                        })
                    elif wl_min > max_wavelength and wl_min <= max_wavelength + wavelength_margin:
                        near_misses['wavelength_max'].append({
                            'id': obs.get('obs_id', 'unknown'),
                            'value': wl_min,
                            'threshold': max_wavelength,
                            'margin': wavelength_margin
                        })
            except (ValueError, TypeError, IndexError):
                pass
    
    return dict(near_misses)


def analyze_parameter_distribution(observations, param_name, extraction_func=None):
    """
    Analyze the distribution of a parameter across observations
    
    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    param_name : str
        Name of the parameter to analyze
    extraction_func : callable or None
        Function to extract parameter from observation, 
        if None uses direct dictionary lookup
        
    Returns:
    --------
    dict : Statistics about the parameter distribution
    """
    values = []
    
    for obs in observations:
        try:
            if extraction_func:
                value = extraction_func(obs)
            else:
                value = obs.get(param_name)
                
            if value is not None:
                values.append(value)
        except (ValueError, TypeError, KeyError):
            pass
    
    if not values:
        return {'count': 0}
    
    # Determine value type and compute appropriate statistics
    if all(isinstance(v, (int, float)) for v in values):
        # Numeric data
        np_values = np.array(values)
        
        return {
            'count': len(values),
            'min': np.min(np_values),
            'max': np.max(np_values),
            'mean': np.mean(np_values),
            'median': np.median(np_values),
            'std': np.std(np_values),
            'percentiles': {
                '5': np.percentile(np_values, 5),
                '25': np.percentile(np_values, 25),
                '75': np.percentile(np_values, 75),
                '95': np.percentile(np_values, 95)
            }
        }
    else:
        # Categorical data
        counter = Counter(values)
        
        return {
            'count': len(values),
            'unique_count': len(counter),
            'top_values': counter.most_common(10)
        }


def extract_ecliptic_latitude(observation):
    """
    Extract ecliptic latitude from an observation
    
    Parameters:
    -----------
    observation : dict
        Observation dictionary
        
    Returns:
    --------
    float : Ecliptic latitude in degrees, or None if not calculable
    """
    # Find coordinate fields
    for ra_field, dec_field in [('s_ra', 's_dec'), ('ra', 'dec'), ('RA', 'DEC')]:
        if ra_field in observation and dec_field in observation:
            try:
                ra = float(observation[ra_field])
                dec = float(observation[dec_field])
                
                # Calculate ecliptic latitude
                coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
                ecliptic_coord = coord.transform_to('ecliptic')
                return ecliptic_coord.lat.deg
            except (ValueError, TypeError):
                pass
    
    return None


def extract_wavelength_center(observation):
    """
    Extract central wavelength from an observation
    
    Parameters:
    -----------
    observation : dict
        Observation dictionary
        
    Returns:
    --------
    float : Central wavelength in microns, or None if not available
    """
    if 'wavelength_range' in observation and observation['wavelength_range']:
        try:
            wavelength = observation['wavelength_range']
            if isinstance(wavelength, (list, tuple, np.ndarray)) and len(wavelength) >= 2:
                wl_min, wl_max = wavelength[0], wavelength[1]
                
                # Convert to microns if needed
                if wl_min < 1e-4 or wl_max < 1e-4:  # Likely in meters
                    wl_min *= 1e6
                    wl_max *= 1e6
                
                return (wl_min + wl_max) / 2
        except (ValueError, TypeError, IndexError):
            pass
    
    return None