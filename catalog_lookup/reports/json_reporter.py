"""
json_reporter.py - Creates machine-readable JSON outputs for KBO candidates

This module creates structured JSON reports for KBO candidate detections,
providing machine-readable outputs suitable for programmatic consumption
and further processing.
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np
from astropy.time import Time

# Set up logging
logger = logging.getLogger(__name__)

class JSONReporter:
    """
    Creates machine-readable JSON outputs for KBO candidates.
    
    This class provides methods to generate structured JSON reports for
    KBO detection results, object matches, and catalog lookups.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the JSON reporter.
        
        Args:
            output_dir: Directory to save JSON reports to. If None, reports are returned as strings.
        """
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
    
    def generate_candidate_report(self, 
                               candidate: Dict[str, Any],
                               matches: Dict[str, Any],
                               catalog_results: Dict[str, Any] = None) -> str:
        """
        Generate a JSON report for a KBO candidate.
        
        Args:
            candidate: Dictionary with candidate data.
            matches: Dictionary with match results from query_manager.
            catalog_results: Optional detailed catalog query results.
            
        Returns:
            JSON string representation of the report, or path to saved file.
        """
        # Create the report structure
        report = {
            "report_type": "kbo_candidate",
            "timestamp": datetime.now().isoformat(),
            "candidate": self._sanitize_for_json(candidate),
            "match_results": self._sanitize_for_json(matches)
        }
        
        # Add catalog results if provided (but limit size)
        if catalog_results:
            # Include only summary information to keep report size manageable
            report["catalog_summary"] = self._create_catalog_summary(catalog_results)
        
        # Generate JSON
        report_json = json.dumps(report, indent=2)
        
        # Save to file if output directory specified
        if self.output_dir:
            # Create a unique filename based on candidate ID and timestamp
            candidate_id = candidate.get("id", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kbo_candidate_{candidate_id}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(report_json)
            
            logger.info(f"Saved candidate report to {filepath}")
            return filepath
        
        # Otherwise return the JSON string
        return report_json
    
    def generate_detection_session_report(self, 
                                       session_info: Dict[str, Any], 
                                       candidates: List[Dict[str, Any]]) -> str:
        """
        Generate a JSON report for an entire KBO detection session.
        
        Args:
            session_info: Dictionary with detection session information.
            candidates: List of dictionaries with candidate data.
            
        Returns:
            JSON string representation of the report, or path to saved file.
        """
        # Create the report structure
        report = {
            "report_type": "detection_session",
            "timestamp": datetime.now().isoformat(),
            "session_info": self._sanitize_for_json(session_info),
            "candidates_summary": {
                "total_candidates": len(candidates),
                "likely_new_objects": sum(1 for c in candidates if c.get("classification") == "possible_new"),
                "known_matches": sum(1 for c in candidates if c.get("classification", "").startswith("known_")),
                "uncertain": sum(1 for c in candidates if c.get("classification") == "uncertain")
            },
            "candidates": [self._sanitize_for_json(c) for c in candidates]
        }
        
        # Generate JSON
        report_json = json.dumps(report, indent=2)
        
        # Save to file if output directory specified
        if self.output_dir:
            # Create a filename based on session info
            session_id = session_info.get("id", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_session_{session_id}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(report_json)
            
            logger.info(f"Saved detection session report to {filepath}")
            return filepath
        
        # Otherwise return the JSON string
        return report_json
    
    def generate_catalog_lookup_report(self, 
                                     query_info: Dict[str, Any], 
                                     results: Dict[str, Any]) -> str:
        """
        Generate a JSON report for catalog lookup results.
        
        Args:
            query_info: Dictionary with query information.
            results: Dictionary with catalog lookup results.
            
        Returns:
            JSON string representation of the report, or path to saved file.
        """
        # Create the report structure
        report = {
            "report_type": "catalog_lookup",
            "timestamp": datetime.now().isoformat(),
            "query_info": self._sanitize_for_json(query_info),
            "results": self._sanitize_for_json(results)
        }
        
        # Generate JSON
        report_json = json.dumps(report, indent=2)
        
        # Save to file if output directory specified
        if self.output_dir:
            # Create a filename based on query info
            query_type = query_info.get("type", "coords")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"catalog_lookup_{query_type}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(report_json)
            
            logger.info(f"Saved catalog lookup report to {filepath}")
            return filepath
        
        # Otherwise return the JSON string
        return report_json
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        Sanitize Python objects to be JSON-serializable.
        
        Args:
            obj: Python object to sanitize.
            
        Returns:
            JSON-serializable object.
        """
        if obj is None:
            return None
        
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        if isinstance(obj, (datetime, Time)):
            return obj.isoformat()
        
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
            
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
            
        if isinstance(obj, np.bool_):
            return bool(obj)
            
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
            
        if isinstance(obj, np.ndarray):
            return [self._sanitize_for_json(item) for item in obj.tolist()]
            
        if isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
            
        # Try object's __dict__
        if hasattr(obj, "__dict__"):
            return self._sanitize_for_json(obj.__dict__)
            
        # Last resort: convert to string
        return str(obj)
    
    def _create_catalog_summary(self, catalog_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of catalog lookup results.
        
        Args:
            catalog_results: Raw catalog results.
            
        Returns:
            Dictionary with summarized catalog information.
        """
        summary = {}
        
        # Process each catalog's results
        for catalog, results in catalog_results.items():
            if catalog == "combined":
                continue  # Skip combined results
                
            catalog_summary = {
                "query_successful": "error" not in results,
                "total_objects": 0
            }
            
            # Count objects based on catalog-specific structure
            if catalog == "mpc":
                if "objects" in results:
                    catalog_summary["total_objects"] = len(results["objects"])
            
            elif catalog == "jpl":
                if "data" in results:
                    catalog_summary["total_objects"] = len(results["data"])
            
            elif catalog == "skybot":
                if "data" in results:
                    catalog_summary["total_objects"] = len(results["data"])
            
            elif catalog == "panstarrs":
                if "data" in results:
                    catalog_summary["total_objects"] = len(results["data"])
            
            elif catalog == "ossos":
                if isinstance(results, list):
                    catalog_summary["total_objects"] = len(results)
            
            # Add to summary
            summary[catalog] = catalog_summary
        
        return summary