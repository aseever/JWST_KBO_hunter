"""
match_evaluator.py - Evaluates match quality between KBO candidates and known objects

This module provides advanced functionality for determining if a detected KBO
candidate matches a known object, with detailed scoring, probability estimation,
and classification based on orbital parameters and position measurements.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import logging
from scipy.stats import norm
from dataclasses import dataclass

# Import utilities
from catalog_lookup.utils.coordinates import (
    calculate_separation, degrees_to_hms, estimate_rate_of_motion
)

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Class to store match evaluation results."""
    is_match: bool
    confidence: float
    match_catalog: Optional[str] = None
    match_object: Optional[Dict[str, Any]] = None
    separation: Optional[float] = None
    proper_motion_match: Optional[bool] = None
    matches: Optional[List[Dict[str, Any]]] = None
    classification: str = "unknown"
    notes: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize optional fields if not provided."""
        if self.notes is None:
            self.notes = []
        if self.matches is None:
            self.matches = []

class MatchEvaluator:
    """
    Evaluates matches between KBO candidates and known objects.
    
    This class provides methods to assess whether a detected KBO candidate
    matches a known object from astronomical catalogs, with detailed
    scoring and confidence estimation.
    """
    
    # Classification thresholds
    MATCH_THRESHOLDS = {
        'high_confidence': 0.9,   # 90% confidence for high-confidence match
        'medium_confidence': 0.7,  # 70% confidence for medium-confidence match
        'low_confidence': 0.4,    # 40% confidence for low-confidence
        'position_tolerance': 10.0,  # arcseconds
        'proper_motion_tolerance': 20.0  # percent
    }
    
    def __init__(self, 
                 position_tolerance_arcsec: float = 10.0,
                 proper_motion_tolerance_percent: float = 20.0,
                 position_weight: float = 0.7,
                 proper_motion_weight: float = 0.3,
                 verbose: bool = False):
        """
        Initialize the match evaluator.
        
        Args:
            position_tolerance_arcsec: Positional tolerance in arcseconds.
            proper_motion_tolerance_percent: Proper motion tolerance as percentage.
            position_weight: Weight for position match in overall score.
            proper_motion_weight: Weight for proper motion match in overall score.
            verbose: Whether to log verbose output.
        """
        self.position_tolerance_arcsec = position_tolerance_arcsec
        self.proper_motion_tolerance_percent = proper_motion_tolerance_percent
        self.position_weight = position_weight
        self.proper_motion_weight = proper_motion_weight
        self.verbose = verbose
    
    def evaluate_match(self, 
                      candidate: Dict[str, Any], 
                      known_object: Dict[str, Any],
                      known_catalog: str) -> MatchResult:
        """
        Evaluate the quality of a match between a candidate and a known object.
        
        Args:
            candidate: Dictionary with candidate object data.
            known_object: Dictionary with known object data.
            known_catalog: Name of the catalog for the known object.
            
        Returns:
            MatchResult object with match evaluation details.
        """
        # Extract coordinates
        if 'ra' not in candidate or 'dec' not in candidate:
            return MatchResult(
                is_match=False,
                confidence=0.0,
                notes=["Candidate lacks position information"]
            )
            
        if 'ra' not in known_object or 'dec' not in known_object:
            return MatchResult(
                is_match=False,
                confidence=0.0,
                notes=["Known object lacks position information"]
            )
        
        # Calculate position score
        position_score = self._calculate_position_score(candidate, known_object)
        if position_score == 0.0:
            return MatchResult(
                is_match=False,
                confidence=0.0,
                match_catalog=known_catalog,
                match_object=known_object,
                separation=self._calculate_separation(candidate, known_object),
                notes=["Position difference exceeds tolerance"]
            )
        
        # Calculate proper motion score if available
        proper_motion_score = self._calculate_proper_motion_score(candidate, known_object)
        proper_motion_match = proper_motion_score > 0.5 if proper_motion_score is not None else None
        
        # Calculate overall confidence
        if proper_motion_score is not None:
            confidence = (
                self.position_weight * position_score + 
                self.proper_motion_weight * proper_motion_score
            )
        else:
            confidence = position_score
        
        # Determine if it's a match based on confidence
        is_match = confidence >= self.MATCH_THRESHOLDS['low_confidence']
        
        # Determine classification
        if confidence >= self.MATCH_THRESHOLDS['high_confidence']:
            classification = "known_high_confidence"
        elif confidence >= self.MATCH_THRESHOLDS['medium_confidence']:
            classification = "known_medium_confidence"
        elif confidence >= self.MATCH_THRESHOLDS['low_confidence']:
            classification = "known_low_confidence"
        else:
            classification = "possible_new"
        
        # Create notes about the match
        notes = []
        separation = self._calculate_separation(candidate, known_object)
        if separation is not None:
            notes.append(f"Position separation: {separation:.3f} arcsec")
        
        if proper_motion_score is not None:
            notes.append(f"Proper motion match: {proper_motion_score:.2f}")
        
        # Create match result
        return MatchResult(
            is_match=is_match,
            confidence=confidence,
            match_catalog=known_catalog,
            match_object=known_object,
            separation=separation,
            proper_motion_match=proper_motion_match,
            classification=classification,
            notes=notes
        )
    
    def evaluate_candidate(self, 
                         candidate: Dict[str, Any],
                         potential_matches: Dict[str, List[Dict[str, Any]]]) -> MatchResult:
        """
        Evaluate a candidate against multiple potential matches from different catalogs.
        
        Args:
            candidate: Dictionary with candidate object data.
            potential_matches: Dictionary mapping catalog names to lists of potential matches.
            
        Returns:
            MatchResult object with best match and overall evaluation.
        """
        all_matches = []
        match_results = []
        
        # Evaluate each potential match
        for catalog, matches in potential_matches.items():
            for known_object in matches:
                match_result = self.evaluate_match(candidate, known_object, catalog)
                if match_result.is_match:
                    match_results.append(match_result)
                    all_matches.append({
                        'catalog': catalog,
                        'object': known_object,
                        'confidence': match_result.confidence,
                        'separation': match_result.separation,
                        'proper_motion_match': match_result.proper_motion_match
                    })
        
        # Sort matches by confidence
        all_matches.sort(key=lambda x: x['confidence'], reverse=True)
        match_results.sort(key=lambda x: x.confidence, reverse=True)
        
        # If no matches found, return result indicating possible new object
        if not match_results:
            return MatchResult(
                is_match=False,
                confidence=0.0,
                classification="possible_new",
                matches=all_matches,
                notes=["No matching known objects found"]
            )
        
        # Get best match
        best_match = match_results[0]
        
        # Check for multiple high-confidence matches
        high_confidence_matches = [m for m in match_results 
                                 if m.confidence >= self.MATCH_THRESHOLDS['high_confidence']]
        
        notes = list(best_match.notes)
        
        if len(high_confidence_matches) > 1:
            notes.append(f"Multiple high-confidence matches found ({len(high_confidence_matches)})")
            notes.append("Best match used for classification")
        
        # Create result using best match properties
        return MatchResult(
            is_match=best_match.is_match,
            confidence=best_match.confidence,
            match_catalog=best_match.match_catalog,
            match_object=best_match.match_object,
            separation=best_match.separation,
            proper_motion_match=best_match.proper_motion_match,
            classification=best_match.classification,
            matches=all_matches,
            notes=notes
        )
    
    def classify_from_multiple_detections(self, 
                                        detections: List[Dict[str, Any]],
                                        potential_matches: Dict[str, List[Dict[str, Any]]]) -> MatchResult:
        """
        Classify a candidate based on multiple detections over time.
        
        Args:
            detections: List of dictionaries with detection data over time.
            potential_matches: Dictionary mapping catalog names to lists of potential matches.
            
        Returns:
            MatchResult object with classification based on multiple detections.
        """
        if not detections:
            return MatchResult(
                is_match=False,
                confidence=0.0,
                classification="insufficient_data",
                notes=["No detection data provided"]
            )
        
        # Evaluate each detection against potential matches
        detection_results = []
        for detection in detections:
            result = self.evaluate_candidate(detection, potential_matches)
            detection_results.append(result)
        
        # Analyze consistency of matches across detections
        match_consistency = self._analyze_match_consistency(detection_results)
        
        # Use the most recently (last) detection result as the base
        latest_result = detection_results[-1]
        
        # Update classification based on consistency analysis
        classification = latest_result.classification
        confidence = latest_result.confidence
        
        # If we have consistent matches across multiple detections, increase confidence
        if match_consistency.get('consistent_matches', False):
            # Boost confidence but cap at 0.99
            confidence = min(0.99, confidence + 0.1 * len(detections))
            
            # Update classification based on new confidence
            if confidence >= self.MATCH_THRESHOLDS['high_confidence']:
                classification = "known_high_confidence"
            elif confidence >= self.MATCH_THRESHOLDS['medium_confidence']:
                classification = "known_medium_confidence"
        
        # If we have inconsistent matches, this might be suspicious
        if match_consistency.get('inconsistent_matches', False):
            classification = "inconsistent_matches"
            
        # Create notes combining latest result and consistency analysis
        notes = list(latest_result.notes)
        if 'notes' in match_consistency:
            notes.extend(match_consistency['notes'])
        
        # Create result
        return MatchResult(
            is_match=latest_result.is_match,
            confidence=confidence,
            match_catalog=latest_result.match_catalog,
            match_object=latest_result.match_object,
            separation=latest_result.separation,
            proper_motion_match=latest_result.proper_motion_match,
            classification=classification,
            matches=latest_result.matches,
            notes=notes
        )
    
    def estimate_new_object_probability(self, 
                                      candidate: Dict[str, Any],
                                      matches: List[Dict[str, Any]],
                                      distance_estimate_au: Optional[float] = None) -> float:
        """
        Estimate the probability that a candidate is a new (previously unknown) object.
        
        Args:
            candidate: Dictionary with candidate object data.
            matches: List of dictionaries with match data.
            distance_estimate_au: Estimated distance in AU, if available.
            
        Returns:
            Estimated probability (0-1) that the object is new.
        """
        # If there are high-confidence matches, the object is likely known
        high_confidence_matches = [m for m in matches 
                                if m['confidence'] >= self.MATCH_THRESHOLDS['high_confidence']]
        if high_confidence_matches:
            return 0.05  # 5% chance of being new (accounting for small uncertainty)
        
        # If there are medium-confidence matches, moderate probability of being new
        medium_confidence_matches = [m for m in matches 
                                   if (m['confidence'] >= self.MATCH_THRESHOLDS['medium_confidence'] and
                                       m['confidence'] < self.MATCH_THRESHOLDS['high_confidence'])]
        if medium_confidence_matches:
            return 0.3  # 30% chance of being new
        
        # If there are only low-confidence matches, higher probability of being new
        low_confidence_matches = [m for m in matches 
                                if (m['confidence'] >= self.MATCH_THRESHOLDS['low_confidence'] and
                                    m['confidence'] < self.MATCH_THRESHOLDS['medium_confidence'])]
        if low_confidence_matches:
            return 0.7  # 70% chance of being new
        
        # If no matches at all, high probability of being new
        if not matches:
            # Adjust based on distance if provided
            if distance_estimate_au is not None:
                if distance_estimate_au < 30:
                    # If closer than 30 AU, less likely to be a previously undiscovered KBO
                    return 0.8  # 80% chance of being new
                elif distance_estimate_au > 50:
                    # If beyond 50 AU, more likely to be a previously undiscovered KBO
                    return 0.95  # 95% chance of being new
            return 0.9  # 90% chance of being new by default
        
        # Fallback case
        return 0.5  # 50% uncertainty
    
    def generate_report(self, 
                      candidate: Dict[str, Any],
                      match_result: MatchResult,
                      include_full_details: bool = False) -> Dict[str, Any]:
        """
        Generate a detailed report for a candidate match evaluation.
        
        Args:
            candidate: Dictionary with candidate object data.
            match_result: MatchResult from evaluation.
            include_full_details: Whether to include detailed match information.
            
        Returns:
            Dictionary with detailed match report.
        """
        report = {
            'candidate': {
                'id': candidate.get('id', 'Unknown'),
                'ra': candidate.get('ra'),
                'dec': candidate.get('dec'),
                'epoch': candidate.get('epoch'),
                'motion_rate': candidate.get('motion_rate'),
                'motion_angle': candidate.get('motion_angle')
            },
            'match_summary': {
                'is_match': match_result.is_match,
                'confidence': match_result.confidence,
                'classification': match_result.classification,
                'separation_arcsec': match_result.separation,
                'proper_motion_match': match_result.proper_motion_match,
                'match_catalog': match_result.match_catalog,
                'new_object_probability': self.estimate_new_object_probability(
                    candidate, match_result.matches)
            },
            'notes': match_result.notes
        }
        
        # Add best match details if available
        if match_result.match_object is not None:
            best_match = match_result.match_object
            report['best_match'] = {
                'id': best_match.get('id', 'Unknown'),
                'name': best_match.get('name', 'Unknown'),
                'catalog': match_result.match_catalog,
                'ra': best_match.get('ra'),
                'dec': best_match.get('dec'),
                'separation_arcsec': match_result.separation
            }
            
            # Add orbital elements if available
            for elem in ['a', 'e', 'i']:
                if elem in best_match:
                    report['best_match'][elem] = best_match[elem]
        
        # Include detailed match information if requested
        if include_full_details and match_result.matches:
            report['all_matches'] = match_result.matches
        
        return report
    
    def _calculate_position_score(self, 
                               candidate: Dict[str, Any], 
                               known_object: Dict[str, Any]) -> float:
        """
        Calculate the position match score based on separation.
        
        Args:
            candidate: Dictionary with candidate object data.
            known_object: Dictionary with known object data.
            
        Returns:
            Score between 0.0 and 1.0, where 1.0 is a perfect match.
        """
        separation_arcsec = self._calculate_separation(candidate, known_object)
        if separation_arcsec is None:
            return 0.0
            
        # If separation exceeds tolerance, return 0
        if separation_arcsec > self.position_tolerance_arcsec:
            return 0.0
            
        # Calculate score using gaussian falloff
        # Score will be 1.0 at 0 arcsec and decrease as separation increases
        # Tolerance is set at 3-sigma point for the gaussian
        sigma = self.position_tolerance_arcsec / 3.0
        score = np.exp(-(separation_arcsec**2) / (2 * sigma**2))
        
        return score
    
    def _calculate_proper_motion_score(self, 
                                    candidate: Dict[str, Any], 
                                    known_object: Dict[str, Any]) -> Optional[float]:
        """
        Calculate the proper motion match score.
        
        Args:
            candidate: Dictionary with candidate object data.
            known_object: Dictionary with known object data.
            
        Returns:
            Score between 0.0 and 1.0, or None if proper motion data unavailable.
        """
        # Check if both objects have proper motion data
        if ('motion_rate' not in candidate or
            'motion_angle' not in candidate or
            'motion_rate' not in known_object or
            'motion_angle' not in known_object):
            return None
            
        # Extract proper motion values
        candidate_rate = candidate['motion_rate']
        candidate_angle = candidate['motion_angle']
        known_rate = known_object['motion_rate']
        known_angle = known_object['motion_angle']
        
        # Normalize angles to 0-360
        candidate_angle = candidate_angle % 360
        known_angle = known_angle % 360
        
        # Calculate angle difference (handling the wrap around 0/360)
        angle_diff = min(abs(candidate_angle - known_angle),
                        360 - abs(candidate_angle - known_angle))
        
        # Calculate rate difference as percentage
        rate_diff_percent = abs(candidate_rate - known_rate) / known_rate * 100
        
        # Check if differences exceed tolerances
        if (rate_diff_percent > self.proper_motion_tolerance_percent or
            angle_diff > 20):  # 20 degrees tolerance for angle
            return 0.0
            
        # Calculate score components
        rate_score = max(0, 1 - rate_diff_percent / self.proper_motion_tolerance_percent)
        angle_score = max(0, 1 - angle_diff / 20)
        
        # Combined score (equal weight to rate and angle)
        score = (rate_score + angle_score) / 2
        
        return score
    
    def _calculate_separation(self, 
                           obj1: Dict[str, Any], 
                           obj2: Dict[str, Any]) -> Optional[float]:
        """
        Calculate separation between two objects in arcseconds.
        
        Args:
            obj1: Dictionary with first object data.
            obj2: Dictionary with second object data.
            
        Returns:
            Separation in arcseconds, or None if coordinates unavailable.
        """
        if ('ra' not in obj1 or 'dec' not in obj1 or
            'ra' not in obj2 or 'dec' not in obj2):
            return None
            
        # Calculate separation in degrees
        separation_deg = calculate_separation(
            obj1['ra'], obj1['dec'], obj2['ra'], obj2['dec'])
            
        # Convert to arcseconds
        separation_arcsec = separation_deg * 3600.0
        
        return separation_arcsec
    
    def _analyze_match_consistency(self, results: List[MatchResult]) -> Dict[str, Any]:
        """
        Analyze consistency of matches across multiple detections.
        
        Args:
            results: List of MatchResult objects from multiple detections.
            
        Returns:
            Dictionary with consistency analysis.
        """
        if not results:
            return {'consistent_matches': False, 'notes': ["No results to analyze"]}
            
        # Extract best matches from each result
        best_matches = []
        for result in results:
            if result.match_object is not None:
                best_matches.append({
                    'catalog': result.match_catalog,
                    'id': result.match_object.get('id', 'Unknown'),
                    'confidence': result.confidence
                })
            else:
                best_matches.append(None)
        
        # Check if all detections match the same object
        consistent_object = True
        
        # Group by object ID and catalog
        match_counts = {}
        for match in best_matches:
            if match is not None:
                key = (match['catalog'], match['id'])
                match_counts[key] = match_counts.get(key, 0) + 1
        
        # Find the most common match
        most_common_match = None
        max_count = 0
        
        for key, count in match_counts.items():
            if count > max_count:
                max_count = count
                most_common_match = key
        
        # Check consistency and create notes
        notes = []
        
        if most_common_match is None:
            consistent_matches = False
            notes.append("No consistent matches found across detections")
        elif max_count == len(results):
            consistent_matches = True
            catalog, obj_id = most_common_match
            notes.append(f"All detections match the same object: {obj_id} in {catalog}")
        elif max_count >= len(results) * 0.75:
            consistent_matches = True
            catalog, obj_id = most_common_match
            notes.append(f"Most detections ({max_count}/{len(results)}) match the same object: {obj_id} in {catalog}")
        else:
            consistent_matches = False
            notes.append(f"Inconsistent matches across detections")
            
        # Check for inconsistent confidence levels
        confidences = [r.confidence for r in results if r.confidence is not None]
        if confidences:
            confidence_range = max(confidences) - min(confidences)
            if confidence_range > 0.4:  # Large confidence variation
                notes.append(f"Large variation in match confidence across detections: {confidence_range:.2f} range")
        
        return {
            'consistent_matches': consistent_matches,
            'inconsistent_matches': not consistent_matches,
            'match_counts': match_counts,
            'most_common_match': most_common_match,
            'most_common_count': max_count,
            'total_detections': len(results),
            'notes': notes
        }