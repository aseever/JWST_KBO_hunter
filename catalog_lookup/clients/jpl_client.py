"""
jpl_client.py - Client for querying JPL's Small-Body Database API

This module provides a client for the JPL SBDB API to search for known
small bodies in the solar system, including Kuiper Belt Objects.

References:
- JPL SBDB API documentation: https://ssd-api.jpl.nasa.gov/doc/sbdb.html
"""

import requests
import logging
from urllib.parse import urljoin
from typing import Dict, Any, List, Optional, Union
import time

# Set up logging
logger = logging.getLogger(__name__)

class JPLClient:
    """
    Client for the JPL Small-Body Database (SBDB) API.
    
    This client allows searching for known small solar system bodies by
    designation, name, coordinates, and other criteria.
    """
    
    # Base URL for the JPL SBDB API
    BASE_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize the JPL SBDB client.
        
        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
        self.session = requests.Session()
    
    def search_by_designation(self, 
                             designation: str, 
                             full_precision: bool = False,
                             phys_par: bool = False,
                             discovery: bool = False) -> Dict[str, Any]:
        """
        Search for an object by its designation or name.
        
        Args:
            designation: Object designation (e.g., "2014 MU69", "Arrokoth")
            full_precision: If True, return full precision values
            phys_par: If True, include physical parameters
            discovery: If True, include discovery information
            
        Returns:
            Dictionary containing object information.
        """
        params = {
            "sstr": designation,
            "full-prec": "true" if full_precision else "false",
            "phys-par": "true" if phys_par else "false",
            "discovery": "true" if discovery else "false",
        }
        
        return self._make_request(params)
    
    def search_by_spk_id(self, spk_id: str) -> Dict[str, Any]:
        """
        Search for an object by its SPK ID.
        
        Args:
            spk_id: SPK ID of the object (e.g., "3788040")
            
        Returns:
            Dictionary containing object information.
        """
        params = {
            "spk": spk_id
        }
        
        return self._make_request(params)
    
    def search_by_constraints(self, 
                             object_type: Optional[str] = None,
                             neo: Optional[bool] = None,
                             pha: Optional[bool] = None,
                             kind: Optional[str] = None,
                             spk: Optional[str] = None,
                             limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Search for objects matching specific constraints.
        
        Args:
            object_type: Type of object ('ast' for asteroid, 'com' for comet)
            neo: If True, limit to Near-Earth Objects
            pha: If True, limit to Potentially Hazardous Asteroids
            kind: Object kind (e.g., 'TNO' for Trans-Neptunian Object)
            spk: SPK ID or range (e.g., "3788040" or "3788000-3789000")
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing search results.
        """
        params = {}
        
        if object_type:
            params["object-type"] = object_type
        if neo is not None:
            params["neo"] = "true" if neo else "false"
        if pha is not None:
            params["pha"] = "true" if pha else "false"
        if kind:
            params["kind"] = kind
        if spk:
            params["spk"] = spk
        if limit:
            params["limit"] = limit
        
        return self._make_request(params)
    
    def search_for_kuiper_belt_objects(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Search specifically for Kuiper Belt Objects.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing search results.
        """
        return self.search_by_constraints(kind="TNO", limit=limit)
    
    def get_close_approaches(self, 
                            designation: str, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Get close approach data for an object.
        
        Args:
            designation: Object designation (e.g., "2014 MU69")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing close approach data.
        """
        params = {
            "sstr": designation,
            "ca-data": "true"
        }
        
        if start_date:
            params["date-min"] = start_date
        if end_date:
            params["date-max"] = end_date
        if limit:
            params["limit"] = limit
        
        return self._make_request(params)
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the JPL SBDB API.
        
        Args:
            params: Request parameters.
            
        Returns:
            Dictionary containing the JSON response.
            
        Raises:
            requests.HTTPError: If the request fails.
        """
        params["format"] = "json"  # Always request JSON response
        
        try:
            logger.debug(f"Making request to {self.BASE_URL} with params {params}")
            response = self.session.get(
                self.BASE_URL, 
                params=params, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
                
        except requests.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
        except ValueError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return {"error": "Unable to parse response", "raw_response": response.text}

    def batch_search(self, 
                    designations: List[str],
                    rate_limit_sleep: float = 1.0) -> List[Dict[str, Any]]:
        """
        Search for multiple objects by designation.
        
        Args:
            designations: List of object designations.
            rate_limit_sleep: Time to sleep between requests to avoid rate limiting.
            
        Returns:
            List of dictionaries containing search results for each designation.
        """
        results = []
        
        for designation in designations:
            result = self.search_by_designation(designation)
            results.append(result)
            
            # Sleep to avoid hitting rate limits
            time.sleep(rate_limit_sleep)
            
        return results