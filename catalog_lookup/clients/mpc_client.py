"""
mpc_client.py - Client for querying the Minor Planet Center's Web Service

This module provides a client for the MPC's web service to search for known
objects near specific coordinates or with specific designations.

References:
- MPC API documentation: https://minorplanetcenter.net/web_service/
"""

import requests
import logging
from urllib.parse import urljoin
from typing import Dict, Any, List, Optional, Union
import time

# Set up logging
logger = logging.getLogger(__name__)

class MPCClient:
    """
    Client for the Minor Planet Center Web Service API.
    
    This client allows searching for known solar system objects by
    coordinates, designation, or other criteria.
    """
    
    # Base URL for the MPC API
    BASE_URL = "https://minorplanetcenter.net/web_service/"
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize the MPC client.
        
        Args:
            api_key: Optional API key for the MPC service. Not required for basic queries.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def search_by_coordinates(self, 
                             ra: float, 
                             dec: float, 
                             radius: float = 1.0,
                             date: Optional[str] = None,
                             limit: int = 100) -> Dict[str, Any]:
        """
        Search for known objects near the specified coordinates.
        
        Args:
            ra: Right ascension in decimal degrees.
            dec: Declination in decimal degrees.
            radius: Search radius in degrees.
            date: Optional observation date (YYYY-MM-DD format).
            limit: Maximum number of results to return.
            
        Returns:
            Dictionary containing search results.
        """
        endpoint = "search"
        params = {
            "ra": ra,
            "dec": dec,
            "radius": radius,
            "limit": limit,
        }
        
        if date:
            params["date"] = date
            
        return self._make_request(endpoint, params)
    
    def search_by_designation(self, designation: str) -> Dict[str, Any]:
        """
        Search for an object by its designation.
        
        Args:
            designation: Object designation (e.g., "2014 MU69" or "486958").
            
        Returns:
            Dictionary containing object information.
        """
        endpoint = "mpc_lookup"
        params = {
            "designation": designation,
            "json": 1  # Request JSON response
        }
        
        return self._make_request(endpoint, params)
    
    def get_orbit(self, designation: str) -> Dict[str, Any]:
        """
        Get orbital elements for an object by its designation.
        
        Args:
            designation: Object designation (e.g., "2014 MU69" or "486958").
            
        Returns:
            Dictionary containing orbital elements.
        """
        endpoint = "mpc_orb"
        params = {
            "designation": designation,
            "json": 1  # Request JSON response
        }
        
        return self._make_request(endpoint, params)
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the MPC API.
        
        Args:
            endpoint: API endpoint.
            params: Request parameters.
            
        Returns:
            Dictionary containing the JSON response.
            
        Raises:
            requests.HTTPError: If the request fails.
        """
        url = urljoin(self.BASE_URL, endpoint)
        
        try:
            logger.debug(f"Making request to {url} with params {params}")
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Check if response is JSON
            if response.headers.get('Content-Type', '').startswith('application/json'):
                return response.json()
            else:
                # Some MPC endpoints return text that needs to be parsed
                try:
                    return {"text_response": response.text}
                except:
                    return {"error": "Unable to parse response", "raw_response": response.text}
                
        except requests.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
        except ValueError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return {"error": "Unable to parse response", "raw_response": response.text}

    def batch_search_coordinates(self, 
                               coordinates: List[Dict[str, float]],
                               radius: float = 1.0,
                               date: Optional[str] = None,
                               rate_limit_sleep: float = 1.0) -> List[Dict[str, Any]]:
        """
        Search for known objects near multiple coordinates.
        
        Args:
            coordinates: List of dictionaries with 'ra' and 'dec' keys (decimal degrees).
            radius: Search radius in degrees.
            date: Optional observation date (YYYY-MM-DD format).
            rate_limit_sleep: Time to sleep between requests to avoid rate limiting.
            
        Returns:
            List of dictionaries containing search results for each coordinate.
        """
        results = []
        
        for coord in coordinates:
            result = self.search_by_coordinates(
                ra=coord['ra'],
                dec=coord['dec'],
                radius=radius,
                date=date
            )
            results.append(result)
            
            # Sleep to avoid hitting rate limits
            time.sleep(rate_limit_sleep)
            
        return results