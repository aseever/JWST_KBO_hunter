"""
skybot_client.py - Client for querying the SkyBoT (Sky Bodies Tracker) service

This module provides a client for the IMCCE SkyBoT service, which can identify 
known solar system objects in a given field of view at a specific time.

References:
- SkyBoT API documentation: http://vo.imcce.fr/webservices/skybot/?documentation
"""

import requests
import logging
from urllib.parse import urljoin
from typing import Dict, Any, List, Optional, Union
import time
from astropy.time import Time
import xml.etree.ElementTree as ET

# Set up logging
logger = logging.getLogger(__name__)

class SkyBotClient:
    """
    Client for the IMCCE SkyBoT (Sky Bodies Tracker) service.
    
    This client allows querying for known solar system objects in a
    specified region of the sky at a given time.
    """
    
    # Base URL for the SkyBoT API
    BASE_URL = "http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize the SkyBoT client.
        
        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
        self.session = requests.Session()
    
    def cone_search(self, 
                   ra: float, 
                   dec: float, 
                   radius: float,
                   epoch: Union[str, Time],
                   location: str = "500",
                   mime_type: str = "json",
                   output_type: str = "object",
                   get_position: bool = True,
                   get_velocity: bool = True,
                   object_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform a cone search for solar system objects around a position.
        
        Args:
            ra: Right ascension in decimal degrees (J2000).
            dec: Declination in decimal degrees (J2000).
            radius: Search radius in degrees.
            epoch: Observation time, either as ISO string or astropy Time object.
            location: Observatory code (default: 500 for geocentric).
            mime_type: Result format: 'json', 'votable', or 'csv'.
            output_type: Output type: 'object' for objects, 'state' for state vectors.
            get_position: Include position in results if True.
            get_velocity: Include velocity in results if True.
            object_type: Filter by object type ('ast' for asteroids, 'com' for comets, etc.).
            
        Returns:
            Dictionary containing search results.
        """
        # Convert epoch to ISO format if it's an astropy Time object
        if isinstance(epoch, Time):
            epoch_str = epoch.iso
        else:
            epoch_str = epoch
            
        params = {
            "RA": ra,
            "DEC": dec,
            "SR": radius,
            "EPOCH": epoch_str,
            "LOC": location,
            "-mime": mime_type,
            "-output": output_type,
        }
        
        # Add optional parameters
        if get_position:
            params["-ep"] = "1"
        if get_velocity:
            params["-vel"] = "1"
        if object_type:
            params["-objtype"] = object_type
        
        return self._make_request(params)
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the SkyBoT API.
        
        Args:
            params: Request parameters.
            
        Returns:
            Dictionary containing the response.
            
        Raises:
            requests.HTTPError: If the request fails.
        """
        try:
            logger.debug(f"Making request to {self.BASE_URL} with params {params}")
            response = self.session.get(
                self.BASE_URL, 
                params=params, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Handle response based on requested mime type
            if params.get("-mime") == "json":
                try:
                    return response.json()
                except ValueError as e:
                    logger.error(f"Error parsing JSON response: {e}")
                    return {"error": "Unable to parse JSON response", "raw_response": response.text}
            
            elif params.get("-mime") == "votable":
                try:
                    # Parse VOTable
                    root = ET.fromstring(response.text)
                    # Convert VOTable to a simplified dict
                    # This is a basic implementation - you might want to use a proper VOTable parser
                    result = {"data": []}
                    for table in root.findall(".//{http://www.ivoa.net/xml/VOTable/v1.3}TABLE"):
                        for resource in table.findall(".//{http://www.ivoa.net/xml/VOTable/v1.3}RESOURCE"):
                            # Extract fields and data
                            fields = []
                            for field in resource.findall(".//{http://www.ivoa.net/xml/VOTable/v1.3}FIELD"):
                                fields.append(field.get("name"))
                            
                            for row in resource.findall(".//{http://www.ivoa.net/xml/VOTable/v1.3}TR"):
                                row_data = {}
                                for i, td in enumerate(row.findall(".//{http://www.ivoa.net/xml/VOTable/v1.3}TD")):
                                    if i < len(fields):
                                        row_data[fields[i]] = td.text
                                result["data"].append(row_data)
                            
                    return result
                except Exception as e:
                    logger.error(f"Error parsing VOTable response: {e}")
                    return {"error": "Unable to parse VOTable response", "raw_response": response.text}
            
            else:
                # For CSV or other formats, return the text
                return {"text_response": response.text}
                
        except requests.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            raise

    def batch_cone_search(self, 
                         coordinates: List[Dict[str, float]],
                         epoch: Union[str, Time],
                         radius: float = 0.5,
                         rate_limit_sleep: float = 1.0) -> List[Dict[str, Any]]:
        """
        Perform cone searches for multiple coordinates.
        
        Args:
            coordinates: List of dictionaries with 'ra' and 'dec' keys.
            epoch: Observation time, either as ISO string or astropy Time object.
            radius: Search radius in degrees.
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
                epoch=epoch
            )
            results.append(result)
            
            # Sleep to avoid hitting rate limits
            time.sleep(rate_limit_sleep)
            
        return results
    
    def search_for_tno(self, 
                     ra: float, 
                     dec: float, 
                     radius: float,
                     epoch: Union[str, Time]) -> Dict[str, Any]:
        """
        Perform a cone search specifically for Trans-Neptunian Objects.
        
        Args:
            ra: Right ascension in decimal degrees (J2000).
            dec: Declination in decimal degrees (J2000).
            radius: Search radius in degrees.
            epoch: Observation time, either as ISO string or astropy Time object.
            
        Returns:
            Dictionary containing search results.
        """
        # SkyBoT doesn't have a direct filter for TNOs, but they're a subset of 
        # asteroids with high semi-major axes, so we'll filter those afterward
        result = self.cone_search(
            ra=ra,
            dec=dec,
            radius=radius,
            epoch=epoch,
            get_position=True,
            get_velocity=True
        )
        
        # Filter for potential TNOs based on properties
        # Note: This is a simplified approach - proper TNO identification would 
        # require looking at orbital elements
        if 'data' in result:
            # Basic TNO filtering logic - this could be improved
            tno_candidates = []
            for obj in result['data']:
                # Look for objects with class indicating TNO or ID indicating TNO
                if ('class' in obj and obj['class'] in ['tno', 'TNO', 'sdo', 'SDO']) or \
                   ('name' in obj and 'TNO' in obj['name']):
                    tno_candidates.append(obj)
            
            result['tno_candidates'] = tno_candidates
            result['num_tno_candidates'] = len(tno_candidates)
        
        return result