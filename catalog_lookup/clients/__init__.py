"""
Catalog API Clients

This module contains client interfaces for various astronomical catalogs
and databases. Each client handles the API communication, data formatting,
and error handling for a specific catalog service.

Available clients:
- MPC: Minor Planet Center database client
- JPL: JPL Small-Body Database client
- SkyBoT: Sky Body Tracker service client
- PanSTARRS: Pan-STARRS Object Catalog client
- OSSOS: Outer Solar System Origins Survey client
"""

# Import clients
from catalog_lookup.clients.mpc_client import MPCClient
from catalog_lookup.clients.jpl_client import JPLClient
from catalog_lookup.clients.skybot_client import SkyBoTClient
from catalog_lookup.clients.panstarrs_client import PanSTARRSClient
from catalog_lookup.clients.ossos_client import OSSOSClient

# Factory function to get the appropriate client
def get_client(catalog_name):
    """
    Get a catalog client by name
    
    Parameters:
    -----------
    catalog_name : str
        Name of the catalog ('mpc', 'jpl', 'skybot', 'panstarrs', 'ossos')
    
    Returns:
    --------
    CatalogClient
        Instance of the requested catalog client
    
    Raises:
    -------
    ValueError
        If catalog_name is not recognized
    """
    catalog_map = {
        'mpc': MPCClient,
        'jpl': JPLClient,
        'skybot': SkyBoTClient,
        'panstarrs': PanSTARRSClient,
        'ossos': OSSOSClient
    }
    
    if catalog_name.lower() not in catalog_map:
        raise ValueError(f"Unknown catalog: {catalog_name}. Available catalogs: {', '.join(catalog_map.keys())}")
    
    return catalog_map[catalog_name.lower()]()

# Define which catalogs are enabled by default
DEFAULT_CATALOGS = ['mpc', 'jpl', 'skybot']

__all__ = [
    'MPCClient',
    'JPLClient', 
    'SkyBoTClient',
    'PanSTARRSClient', 
    'OSSOSClient',
    'get_client',
    'DEFAULT_CATALOGS'
]