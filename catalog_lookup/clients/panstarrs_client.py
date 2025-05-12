"""
panstarrs_client.py - Client for querying the Pan-STARRS catalog

This module provides a client for querying the Pan-STARRS catalog,
which contains observations that might include KBOs and other solar
system objects.

References:
- PS1 MAST page: https://catalogs.mast.stsci.edu/panstarrs/
- PS1 API documentation: https://outerspace.stsci.edu/display/PANSTARRS/
"""

import requests
import logging
from urllib.parse import urljoin
from typing import Dict, Any, List, Optional, Union
import time
import pandas as pd
import io

# Set up logging
logger = logging.getLogger(__name__)

class PanSTARRSClient:
    """
    Client for the Pan-STARRS catalog.
    
    This client allows searching for objects in the Pan-STARRS catalog
    by coordinates, identifier, or other criteria.
    """
    
    # Base URL for the PS1 MAST API
    BASE_URL = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/"
    
    # URL for the PS1 Image Cutout Service
    CUTOUT_URL = "https://ps1images.stsci.edu/cgi-bin/ps1cutouts"
    
    def __init__(self, timeout: int = 60):
        """
        Initialize the Pan-STARRS client.
        
        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
        self.session = requests.Session()
    
    def cone_search(self, 
                   ra: float, 
                   dec: float, 
                   radius: float = 0.1,
                   catalog: str = "mean",
                   columns: Optional[List[str]] = None,
                   max_records: int = 1000,
                   anomaly_filter: bool = True) -> Dict[str, Any]:
        """
        Perform a cone search in the Pan-STARRS catalog.
        
        Args:
            ra: Right ascension in decimal degrees (J2000).
            dec: Declination in decimal degrees (J2000).
            radius: Search radius in degrees.
            catalog: PS1 catalog to query: 'mean', 'stack', or 'detection'.
            columns: Specific columns to retrieve (None for all).
            max_records: Maximum number of records to return.
            anomaly_filter: Filter out likely anomalies if True.
            
        Returns:
            Dictionary containing search results.
        """
        # Set default columns if none provided
        if columns is None:
            if catalog == "mean":
                columns = ["objID", "raMean", "decMean", "nDetections", "ng", "nr", "ni", "nz", "ny",
                          "gMeanPSFMag", "rMeanPSFMag", "iMeanPSFMag", "zMeanPSFMag", "yMeanPSFMag"]
            elif catalog == "stack":
                columns = ["objID", "raStack", "decStack", "ng", "nr", "ni", "nz", "ny",
                          "gPSFMag", "rPSFMag", "iPSFMag", "zPSFMag", "yPSFMag"]
            else:  # detection
                columns = ["objID", "detectID", "ra", "dec", "epoch", "filter", "magnitude", "autoMag", "psfMag"]
        
        # Build the URL
        url = urljoin(self.BASE_URL, f"{catalog}/")
        
        # Set parameters
        params = {
            "ra": ra,
            "dec": dec,
            "radius": radius,
            "max_records": max_records,
            "format": "json"
        }
        
        # Add columns parameter
        if columns:
            params["columns"] = ",".join(columns)
            
        # Add anomaly filter if requested
        if anomaly_filter and catalog == "mean":
            params["anomaly_filter"] = "true"
        
        # Make the request
        return self._make_request(url, params)
    
    def object_search(self, 
                     objid: str, 
                     catalog: str = "mean",
                     columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search for a specific object by objID.
        
        Args:
            objid: Pan-STARRS object ID.
            catalog: PS1 catalog to query: 'mean', 'stack', or 'detection'.
            columns: Specific columns to retrieve (None for all).
            
        Returns:
            Dictionary containing object information.
        """
        # Build the URL
        url = urljoin(self.BASE_URL, f"{catalog}/{objid}")
        
        # Set parameters
        params = {
            "format": "json"
        }
        
        # Add columns parameter
        if columns:
            params["columns"] = ",".join(columns)
        
        # Make the request
        return self._make_request(url, params)
    
    def get_detection_history(self, 
                             objid: str, 
                             columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get detection history for a specific object.
        
        Args:
            objid: Pan-STARRS object ID.
            columns: Specific columns to retrieve (None for all).
            
        Returns:
            Dictionary containing detection history.
        """
        if columns is None:
            columns = ["objID", "detectID", "ra", "dec", "epoch", "filter", "magnitude", "psfMag"]
        
        # Build URL to get detections for this object
        url = urljoin(self.BASE_URL, f"detection/")
        
        # Set parameters
        params = {
            "objid": objid,
            "format": "json"
        }
        
        # Add columns parameter
        if columns:
            params["columns"] = ",".join(columns)
        
        # Make the request
        return self._make_request(url, params)
    
    def search_for_moving_objects(self, 
                                ra: float, 
                                dec: float, 
                                radius: float = 0.1,
                                max_records: int = 1000) -> Dict[str, Any]:
        """
        Search for potential moving objects in Pan-STARRS.
        
        Note: This is a heuristic approach, as Pan-STARRS is not specifically
        designed to track moving objects. We look for objects with fewer
        detections or position discrepancies.
        
        Args:
            ra: Right ascension in decimal degrees (J2000).
            dec: Declination in decimal degrees (J2000).
            radius: Search radius in degrees.
            max_records: Maximum number of records to return.
            
        Returns:
            Dictionary containing potential moving object candidates.
        """
        # First search the mean catalog
        mean_results = self.cone_search(
            ra=ra,
            dec=dec,
            radius=radius,
            catalog="mean",
            max_records=max_records
        )
        
        # If no results, return empty
        if not mean_results.get("data"):
            return {"data": [], "message": "No objects found in search area"}
        
        # Look for potential moving objects based on heuristics
        # 1. Few detections
        # 2. Detections in some bands but not others
        # 3. High position variance
        
        moving_candidates = []
        
        for obj in mean_results.get("data", []):
            # Check number of detections - moving objects often have fewer
            n_detections = obj.get("nDetections", 0)
            
            # Check filter coverage - moving objects might be detected in some filters but not others
            bands = {"g": obj.get("ng", 0), "r": obj.get("nr", 0), "i": obj.get("ni", 0), 
                     "z": obj.get("nz", 0), "y": obj.get("ny", 0)}
            
            detected_bands = sum(1 for count in bands.values() if count > 0)
            
            # Criteria for potential moving objects:
            # - Few detections (less than 10)
            # - Inconsistent band coverage (less than 4 bands)
            is_candidate = (n_detections < 10 or detected_bands < 4)
            
            if is_candidate:
                # Get the detection history to check for motion
                if "objID" in obj:
                    # Add a flag for our moving object heuristic
                    obj["moving_object_candidate"] = True
                    obj["moving_object_reasons"] = []
                    
                    if n_detections < 10:
                        obj["moving_object_reasons"].append(f"Few detections: {n_detections}")
                    if detected_bands < 4:
                        obj["moving_object_reasons"].append(f"Detected in only {detected_bands} bands")
                    
                    moving_candidates.append(obj)
        
        return {
            "data": moving_candidates,
            "message": f"Found {len(moving_candidates)} potential moving object candidates out of {len(mean_results.get('data', []))} objects"
        }
    
    def get_image_cutout_url(self, 
                           ra: float, 
                           dec: float, 
                           size: float = 240,
                           filters: str = "grizy",
                           output_format: str = "fits") -> str:
        """
        Get URL for image cutouts of a location in Pan-STARRS.
        
        Args:
            ra: Right ascension in decimal degrees (J2000).
            dec: Declination in decimal degrees (J2000).
            size: Size of cutout in pixels.
            filters: Filter string (e.g., "grizy" for all filters).
            output_format: Output format ("fits" or "jpg").
            
        Returns:
            URL for the image cutout.
        """
        # Build the URL parameters
        params = {
            "ra": ra,
            "dec": dec,
            "size": size,
            "filters": filters,
            "format": output_format,
            "download": 1
        }
        
        # Convert params to query string
        param_string = "&".join(f"{key}={value}" for key, value in params.items())
        
        # Build the full URL
        cutout_url = f"{self.CUTOUT_URL}?{param_string}"
        
        return cutout_url
    
    def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the Pan-STARRS API.
        
        Args:
            url: API endpoint URL.
            params: Request parameters.
            
        Returns:
            Dictionary containing the response.
            
        Raises:
            requests.HTTPError: If the request fails.
        """
        try:
            logger.debug(f"Making request to {url} with params {params}")
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Check if response is JSON
            try:
                return response.json()
            except ValueError as e:
                logger.error(f"Error parsing JSON response: {e}")
                
                # Try to parse as CSV (sometimes PS1 returns CSV)
                try:
                    df = pd.read_csv(io.StringIO(response.text))
                    return {"data": df.to_dict(orient="records")}
                except:
                    return {"error": "Unable to parse response", "raw_response": response.text}
                
        except requests.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            raise

    def batch_cone_search(self, 
                         coordinates: List[Dict[str, float]],
                         radius: float = 0.1,
                         catalog: str = "mean",
                         columns: Optional[List[str]] = None,
                         rate_limit_sleep: float = 1.0) -> List[Dict[str, Any]]:
        """
        Perform cone searches for multiple coordinates.
        
        Args:
            coordinates: List of dictionaries with 'ra' and 'dec' keys.
            radius: Search radius in degrees.
            catalog: PS1 catalog to query.
            columns: Specific columns to retrieve.
            rate_limit_sleep: Time to sleep between requests to avoid rate limiting.
            
        Returns:
            List of dictionaries containing search results for each coordinate.
        """
        results = []
        
        for coord in coordinates:
            result = self.cone_search(
                ra=coord['ra'],
                dec=coord['dec'],
                radius=radius,
                catalog=catalog,
                columns=columns
            )
            results.append(result)
            
            # Sleep to avoid hitting rate limits
            time.sleep(rate_limit_sleep)
            
        return results