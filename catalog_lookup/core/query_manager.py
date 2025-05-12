"""
query_manager.py - Coordinates queries across multiple KBO catalogs

This module provides a centralized system to query multiple astronomical catalogs
for KBO and other solar system object data, with intelligent parallelization,
caching, rate limiting, and result consolidation.
"""

import logging
import time
import concurrent.futures
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

# Import catalog clients
from catalog_lookup.clients.mpc_client import MPCClient
from catalog_lookup.clients.jpl_client import JPLClient
from catalog_lookup.clients.skybot_client import SkyBotClient
from catalog_lookup.clients.panstarrs_client import PanSTARRSClient
from catalog_lookup.clients.ossos_client import OSSOSClient

# Import utilities
from catalog_lookup.utils.coordinates import (
    degrees_to_hms, hms_to_degrees, calculate_separation,
    format_target_for_mpc, format_target_for_skybot
)
from catalog_lookup.utils.cache import QueryCache, cached, default_cache
from catalog_lookup.utils.rate_limiter import (
    mpc_rate_limiter, jpl_rate_limiter, skybot_rate_limiter, panstarrs_rate_limiter
)

# Set up logging
logger = logging.getLogger(__name__)

class QueryManager:
    """
    Manager for coordinating queries across multiple KBO catalogs.
    
    This class provides methods to search for KBOs and other solar system objects
    across multiple catalogs, consolidating the results and applying intelligent
    caching and rate limiting.
    """
    def __init__(self, 
                cache_dir: Optional[str] = None,
                ossos_data_dir: Optional[str] = None,
                parallel_queries: bool = True,
                max_workers: int = 4,
                timeout: int = 60,
                verbose: bool = False,
                disabled_catalogs: Optional[List[str]] = None):
        """
        Initialize the query manager with catalog clients.
        
        Args:
            cache_dir: Directory for query cache. If None, uses "./cache".
            ossos_data_dir: Directory for OSSOS catalog data. If None, downloads data.
            parallel_queries: Whether to use parallel execution for queries.
            max_workers: Maximum number of worker threads for parallel queries.
            timeout: Request timeout in seconds.
            verbose: Whether to log verbose output.
            disabled_catalogs: List of catalog names to disable API calls for.
                            If None, all catalogs are enabled.
        """
        self.cache_dir = cache_dir
        self.parallel_queries = parallel_queries
        self.max_workers = max_workers
        self.timeout = timeout
        self.verbose = verbose
        
        # Store disabled catalogs
        self.disabled_catalogs = disabled_catalogs or []
        
        # Set up cache
        self.cache = QueryCache(cache_dir=cache_dir)
        
        # Set up clients
        logger.info("Initializing catalog clients...")
        self.mpc_client = MPCClient(timeout=timeout)
        self.jpl_client = JPLClient(timeout=timeout)
        self.skybot_client = SkyBotClient(timeout=timeout)
        self.panstarrs_client = PanSTARRSClient(timeout=timeout)
        
        # OSSOS client may take longer to initialize if it needs to download data
        logger.info("Initializing OSSOS client (may take a moment if downloading data)...")
        self.ossos_client = OSSOSClient(data_dir=ossos_data_dir, 
                                    timeout=timeout, 
                                    download_if_missing=True)
        
        # Initialize result storage
        self.last_query_time = None
        self.last_query_results = {}
        
        # Log which catalogs are disabled
        if self.disabled_catalogs:
            logger.info(f"The following catalogs are disabled: {', '.join(self.disabled_catalogs)}")
        
        logger.info("Query manager initialized successfully")

    
    def search_by_coordinates(self, 
                            ra: float, 
                            dec: float,
                            radius: float = 1.0,
                            epoch: Optional[Union[str, Time]] = None,
                            catalogs: Optional[List[str]] = None,
                            combine_results: bool = True) -> Dict[str, Any]:
        """
        Search for objects near specified coordinates across catalogs.
        
        Args:
            ra: Right ascension in decimal degrees (J2000).
            dec: Declination in decimal degrees (J2000).
            radius: Search radius in degrees.
            epoch: Observation time. If None, uses current time.
            catalogs: List of catalogs to query. If None, queries all.
                    Options: 'mpc', 'jpl', 'skybot', 'panstarrs', 'ossos'.
            combine_results: Whether to combine results into a unified format.
            
        Returns:
            Dictionary with results from each catalog.
        """
        # Process epoch
        if epoch is None:
            epoch = Time.now()
        elif isinstance(epoch, str):
            epoch = Time(epoch)
            
        # Normalize catalogs parameter
        if catalogs is None:
            catalogs = ['mpc', 'jpl', 'skybot', 'panstarrs', 'ossos']
        
        # Apply disabled catalogs filter
        active_catalogs = [cat for cat in catalogs if cat not in self.disabled_catalogs]
        
        # Log which catalogs are being skipped for this query
        skipped_catalogs = [cat for cat in catalogs if cat in self.disabled_catalogs]
        if skipped_catalogs:
            logger.info(f"Skipping disabled catalogs: {', '.join(skipped_catalogs)}")
        
        # Prepare query functions
        query_functions = {}
        
        # Only prepare functions for active catalogs
        if 'mpc' in active_catalogs:
            query_functions['mpc'] = self._query_mpc_coordinates
        if 'jpl' in active_catalogs:
            query_functions['jpl'] = self._query_jpl_coordinates
        if 'skybot' in active_catalogs:
            query_functions['skybot'] = self._query_skybot_coordinates
        if 'panstarrs' in active_catalogs:
            query_functions['panstarrs'] = self._query_panstarrs_coordinates
        if 'ossos' in active_catalogs:
            query_functions['ossos'] = self._query_ossos_coordinates
        
        # Create empty results for disabled catalogs
        results = {cat: {'info': 'Catalog disabled', 'objects': []} for cat in skipped_catalogs}
        
        # Execute queries for active catalogs
        if query_functions:
            if self.parallel_queries and len(query_functions) > 1:
                active_results = self._execute_parallel_queries(
                    query_functions, ra, dec, radius, epoch)
            else:
                active_results = self._execute_sequential_queries(
                    query_functions, ra, dec, radius, epoch)
            
            # Merge results
            results.update(active_results)
        
        # Store query information
        self.last_query_time = datetime.now()
        self.last_query_results = results
        
        # Combine results if requested
        if combine_results:
            combined = self._combine_coordinate_search_results(
                results, ra, dec, radius)
            results['combined'] = combined
        
        return results
        
    def search_by_designation(self, 
                             designation: str,
                             catalogs: Optional[List[str]] = None,
                             combine_results: bool = True) -> Dict[str, Any]:
        """
        Search for an object by designation across catalogs.
        
        Args:
            designation: Object designation (e.g., "2014 MU69").
            catalogs: List of catalogs to query. If None, queries all.
                     Options: 'mpc', 'jpl', 'ossos'.
            combine_results: Whether to combine results into a unified format.
            
        Returns:
            Dictionary with results from each catalog.
        """
        # Normalize catalogs parameter - note that not all catalogs support
        # designation search
        if catalogs is None:
            catalogs = ['mpc', 'jpl', 'ossos']
        
        # Prepare query functions
        query_functions = {}
        if 'mpc' in catalogs:
            query_functions['mpc'] = self._query_mpc_designation
        if 'jpl' in catalogs:
            query_functions['jpl'] = self._query_jpl_designation
        if 'ossos' in catalogs:
            query_functions['ossos'] = self._query_ossos_designation
        
        # Execute queries
        if self.parallel_queries and len(query_functions) > 1:
            results = self._execute_parallel_designation_queries(
                query_functions, designation)
        else:
            results = self._execute_sequential_designation_queries(
                query_functions, designation)
        
        # Store query information
        self.last_query_time = datetime.now()
        self.last_query_results = results
        
        # Combine results if requested
        if combine_results:
            combined = self._combine_designation_search_results(results)
            results['combined'] = combined
        
        return results
    
    def search_for_kbos(self, 
                      ra: float, 
                      dec: float,
                      radius: float = 1.0,
                      epoch: Optional[Union[str, Time]] = None) -> Dict[str, Any]:
        """
        Specialized search for KBOs near specified coordinates.
        
        This method uses optimized parameters and post-processing to
        focus on KBO detection.
        
        Args:
            ra: Right ascension in decimal degrees (J2000).
            dec: Declination in decimal degrees (J2000).
            radius: Search radius in degrees.
            epoch: Observation time. If None, uses current time.
            
        Returns:
            Dictionary with KBO results from catalogs and unified KBO list.
        """
        # Process epoch
        if epoch is None:
            epoch = Time.now()
        elif isinstance(epoch, str):
            epoch = Time(epoch)
        
        # Execute coordinate search across all catalogs
        results = self.search_by_coordinates(
            ra=ra, dec=dec, radius=radius, epoch=epoch, combine_results=False)
        
        # Apply KBO-specific filtering and processing
        kbo_results = {}
        
        # MPC results - filter for likely KBOs
        if 'mpc' in results:
            mpc_data = results['mpc']
            if 'objects' in mpc_data:
                kbo_candidates = []
                for obj in mpc_data['objects']:
                    # Use orbit information to identify likely KBOs
                    # This is a simplified approach - real implementation would be more complex
                    orbit_type = obj.get('orbit_type', '').lower()
                    if any(kw in orbit_type for kw in ['tno', 'kbo', 'kuiper', 'sdo']):
                        kbo_candidates.append(obj)
                kbo_results['mpc'] = {'objects': kbo_candidates}
        
        # JPL results - use TNO kind filter
        if 'jpl' in results:
            # Try to use existing results but reformat as a specialized KBO query
            jpl_data = results['jpl']
            # Extract just the TNOs if we have that info
            if 'data' in jpl_data:
                kbo_candidates = []
                for obj in jpl_data['data']:
                    if obj.get('kind', '').lower() in ['tno', 'kbo', 'sdo']:
                        kbo_candidates.append(obj)
                kbo_results['jpl'] = {'data': kbo_candidates}
            
            # If we didn't get good filtering from the original results,
            # make a specific TNO query
            if 'jpl' not in kbo_results or not kbo_results['jpl'].get('data'):
                try:
                    tno_results = self.jpl_client.search_by_constraints(kind="TNO")
                    kbo_results['jpl'] = tno_results
                except Exception as e:
                    logger.error(f"Error querying JPL for TNOs: {e}")
        
        # SkyBoT results - filter for TNOs
        if 'skybot' in results:
            skybot_results = self.skybot_client.search_for_tno(
                ra=ra, dec=dec, radius=radius, epoch=epoch)
            kbo_results['skybot'] = skybot_results
        
        # OSSOS results - already KBO-specific
        if 'ossos' in results:
            kbo_results['ossos'] = results['ossos']
            
        # PanSTARRS results - filter for potential moving objects
        if 'panstarrs' in results:
            # For PanSTARRS, we need to do additional processing to identify KBO candidates
            # This is very approximate as PanSTARRS isn't specialized for KBOs
            ps_results = self.panstarrs_client.search_for_moving_objects(
                ra=ra, dec=dec, radius=radius)
            kbo_results['panstarrs'] = ps_results
        
        # Combine KBO results
        combined_kbos = self._combine_kbo_results(kbo_results, ra, dec)
        kbo_results['combined'] = combined_kbos
        
        return kbo_results
    
    def match_against_known_objects(self, 
                                  ra: float, 
                                  dec: float, 
                                  epoch: Union[str, Time],
                                  search_radius: float = 1.0,
                                  match_radius: float = 0.05) -> Dict[str, Any]:
        """
        Check if a position matches any known objects in catalogs.
        
        This is useful for evaluating KBO candidates to see if they
        match known objects.
        
        Args:
            ra: Right ascension in decimal degrees (J2000).
            dec: Declination in decimal degrees (J2000).
            epoch: Observation time.
            search_radius: Radius to search for objects in degrees.
            match_radius: Maximum separation to consider a match in degrees.
            
        Returns:
            Dictionary with match results and best matches from each catalog.
        """
        # Process epoch
        if isinstance(epoch, str):
            epoch = Time(epoch)
        
        # Search for objects near the position
        results = self.search_by_coordinates(
            ra=ra, dec=dec, radius=search_radius, epoch=epoch, combine_results=False)
        
        # Process each catalog to find matches
        matches = {}
        for catalog, catalog_results in results.items():
            catalog_matches = self._find_matches_in_catalog(
                catalog, catalog_results, ra, dec, match_radius)
            if catalog_matches:
                matches[catalog] = catalog_matches
        
        # Determine if the position matches any known object
        is_known = any(len(m.get('matches', [])) > 0 for m in matches.values())
        
        # Get best match if any
        best_match = None
        best_separation = float('inf')
        
        for catalog, catalog_matches in matches.items():
            for match in catalog_matches.get('matches', []):
                separation = match.get('separation', float('inf'))
                if separation < best_separation:
                    best_separation = separation
                    best_match = {
                        'catalog': catalog,
                        'match': match
                    }
        
        return {
            'is_known': is_known,
            'best_match': best_match,
            'matches': matches
        }
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about the query cache.
        
        Returns:
            Dictionary with cache statistics.
        """
        return self.cache.get_stats()
    
    def clear_cache(self) -> int:
        """
        Clear the query cache.
        
        Returns:
            Number of cache entries cleared.
        """
        return self.cache.clear()
    
    def _execute_parallel_queries(self, 
                                query_functions: Dict[str, Callable], 
                                ra: float, dec: float, 
                                radius: float, epoch: Time) -> Dict[str, Any]:
        """
        Execute coordinate queries in parallel.
        
        Args:
            query_functions: Dictionary mapping catalog names to query functions.
            ra: Right ascension in decimal degrees.
            dec: Declination in decimal degrees.
            radius: Search radius in degrees.
            epoch: Observation time.
            
        Returns:
            Dictionary with results from each catalog.
        """
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_catalog = {}
            
            # Submit all queries
            for catalog, query_func in query_functions.items():
                future = executor.submit(query_func, ra, dec, radius, epoch)
                future_to_catalog[future] = catalog
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_catalog):
                catalog = future_to_catalog[future]
                try:
                    result = future.result()
                    results[catalog] = result
                except Exception as e:
                    logger.error(f"Error in {catalog} query: {e}")
                    results[catalog] = {'error': str(e)}
        
        return results
    
    def _execute_sequential_queries(self, 
                                  query_functions: Dict[str, Callable], 
                                  ra: float, dec: float, 
                                  radius: float, epoch: Time) -> Dict[str, Any]:
        """
        Execute coordinate queries sequentially.
        
        Args:
            query_functions: Dictionary mapping catalog names to query functions.
            ra: Right ascension in decimal degrees.
            dec: Declination in decimal degrees.
            radius: Search radius in degrees.
            epoch: Observation time.
            
        Returns:
            Dictionary with results from each catalog.
        """
        results = {}
        for catalog, query_func in query_functions.items():
            try:
                result = query_func(ra, dec, radius, epoch)
                results[catalog] = result
            except Exception as e:
                logger.error(f"Error in {catalog} query: {e}")
                results[catalog] = {'error': str(e)}
        
        return results
    
    def _execute_parallel_designation_queries(self, 
                                           query_functions: Dict[str, Callable], 
                                           designation: str) -> Dict[str, Any]:
        """
        Execute designation queries in parallel.
        
        Args:
            query_functions: Dictionary mapping catalog names to query functions.
            designation: Object designation.
            
        Returns:
            Dictionary with results from each catalog.
        """
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_catalog = {}
            
            # Submit all queries
            for catalog, query_func in query_functions.items():
                future = executor.submit(query_func, designation)
                future_to_catalog[future] = catalog
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_catalog):
                catalog = future_to_catalog[future]
                try:
                    result = future.result()
                    results[catalog] = result
                except Exception as e:
                    logger.error(f"Error in {catalog} designation query: {e}")
                    results[catalog] = {'error': str(e)}
        
        return results
    
    def _execute_sequential_designation_queries(self, 
                                             query_functions: Dict[str, Callable], 
                                             designation: str) -> Dict[str, Any]:
        """
        Execute designation queries sequentially.
        
        Args:
            query_functions: Dictionary mapping catalog names to query functions.
            designation: Object designation.
            
        Returns:
            Dictionary with results from each catalog.
        """
        results = {}
        for catalog, query_func in query_functions.items():
            try:
                result = query_func(designation)
                results[catalog] = result
            except Exception as e:
                logger.error(f"Error in {catalog} designation query: {e}")
                results[catalog] = {'error': str(e)}
        
        return results
    
    @cached(default_cache)
    def _query_mpc_coordinates(self, ra: float, dec: float, radius: float, epoch: Time) -> Dict[str, Any]:
        """Query the MPC for objects near coordinates."""
        mpc_rate_limiter.wait_if_needed()
        return self.mpc_client.search_by_coordinates(ra, dec, radius, date=epoch.iso.split('T')[0])
    
    @cached(default_cache)
    def _query_jpl_coordinates(self, ra: float, dec: float, radius: float, epoch: Time) -> Dict[str, Any]:
        """Query the JPL SBDB for objects near coordinates."""
        # JPL doesn't have a direct coordinate search, so we'd need to:
        # 1. Get all KBOs and similar objects
        # 2. Filter by position
        jpl_rate_limiter.wait_if_needed()
        
        # Since JPL doesn't have direct cone search, we'll do a TNO search
        # and filter after (in a production system, you'd have a more 
        # sophisticated approach)
        try:
            # Search for TNOs and SDOs which are likely KBOs
            tno_results = self.jpl_client.search_by_constraints(kind="TNO")
            return tno_results
        except Exception as e:
            logger.error(f"Error in JPL TNO query: {e}")
            return {"error": str(e)}
    
    @cached(default_cache)
    def _query_skybot_coordinates(self, ra: float, dec: float, radius: float, epoch: Time) -> Dict[str, Any]:
        """Query SkyBoT for objects near coordinates."""
        skybot_rate_limiter.wait_if_needed()
        return self.skybot_client.cone_search(ra, dec, radius, epoch)
    
    @cached(default_cache)
    def _query_panstarrs_coordinates(self, ra: float, dec: float, radius: float, epoch: Time) -> Dict[str, Any]:
        """Query Pan-STARRS for objects near coordinates."""
        return self.panstarrs_client.cone_search(ra, dec, radius)
    
    @cached(default_cache)
    def _query_ossos_coordinates(self, ra: float, dec: float, radius: float, epoch: Time) -> Dict[str, Any]:
        """Query OSSOS for objects near coordinates."""
        # Convert pandas DataFrame to dict for JSON serialization
        df = self.ossos_client.cone_search(ra, dec, radius, epoch)
        return df.to_dict(orient='records') if not df.empty else {'objects': []}
    
    @cached(default_cache)
    def _query_mpc_designation(self, designation: str) -> Dict[str, Any]:
        """Query the MPC for an object by designation."""
        mpc_rate_limiter.wait_if_needed()
        return self.mpc_client.search_by_designation(designation)
    
    @cached(default_cache)
    def _query_jpl_designation(self, designation: str) -> Dict[str, Any]:
        """Query the JPL SBDB for an object by designation."""
        jpl_rate_limiter.wait_if_needed()
        return self.jpl_client.search_by_designation(designation)
    
    @cached(default_cache)
    def _query_ossos_designation(self, designation: str) -> Dict[str, Any]:
        """Query OSSOS for an object by designation."""
        # Convert pandas DataFrame to dict for JSON serialization
        df = self.ossos_client.search_by_name(designation)
        return df.to_dict(orient='records') if not df.empty else {'objects': []}
    
    def _combine_coordinate_search_results(self, 
                                         results: Dict[str, Any], 
                                         ra: float,
                                         dec: float,
                                         radius: float) -> Dict[str, Any]:
        """
        Combine results from multiple catalogs into a unified format.
        
        Args:
            results: Dictionary with results from each catalog.
            ra: Search center right ascension.
            dec: Search center declination.
            radius: Search radius.
            
        Returns:
            Dictionary with combined results.
        """
        combined_objects = []
        
        # Process each catalog
        for catalog, catalog_results in results.items():
            if catalog == 'combined':
                continue
                
            # Handle errors
            if 'error' in catalog_results:
                logger.warning(f"Error in {catalog} results: {catalog_results['error']}")
                continue
            
            # Extract objects based on catalog-specific structure
            objects = []
            
            if catalog == 'mpc':
                objects = catalog_results.get('objects', [])
            elif catalog == 'jpl':
                objects = catalog_results.get('data', [])
            elif catalog == 'skybot':
                objects = catalog_results.get('data', [])
            elif catalog == 'panstarrs':
                objects = catalog_results.get('data', [])
            elif catalog == 'ossos':
                objects = catalog_results if isinstance(catalog_results, list) else []
            
            # Convert each object to a common format
            for obj in objects:
                unified_obj = self._convert_to_unified_format(catalog, obj)
                unified_obj['catalog'] = catalog
                
                # Calculate separation from search center if coords available
                if 'ra' in unified_obj and 'dec' in unified_obj:
                    obj_ra = unified_obj['ra']
                    obj_dec = unified_obj['dec']
                    separation = calculate_separation(ra, dec, obj_ra, obj_dec)
                    unified_obj['separation'] = separation
                
                combined_objects.append(unified_obj)
        
        # Sort by separation if available
        combined_objects.sort(key=lambda x: x.get('separation', float('inf')))
        
        return {
            'search': {
                'ra': ra,
                'dec': dec,
                'radius': radius
            },
            'total_objects': len(combined_objects),
            'objects': combined_objects
        }
    
    def _combine_designation_search_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine designation search results from multiple catalogs.
        
        Args:
            results: Dictionary with results from each catalog.
            
        Returns:
            Dictionary with combined results.
        """
        combined_info = {}
        catalogs_with_data = []
        
        # Process each catalog
        for catalog, catalog_results in results.items():
            if catalog == 'combined':
                continue
                
            # Handle errors
            if 'error' in catalog_results:
                logger.warning(f"Error in {catalog} results: {catalog_results['error']}")
                continue
            
            # Check if we have data
            has_data = False
            
            if catalog == 'mpc' and catalog_results:
                has_data = True
                # Extract basic properties
                combined_info.update(self._extract_mpc_object_properties(catalog_results))
                
            elif catalog == 'jpl' and catalog_results.get('data'):
                has_data = True
                # Extract basic properties
                combined_info.update(self._extract_jpl_object_properties(catalog_results))
                
            elif catalog == 'ossos' and catalog_results:
                has_data = True
                # Extract basic properties if we have any data
                if isinstance(catalog_results, list) and catalog_results:
                    combined_info.update(self._extract_ossos_object_properties(catalog_results[0]))
            
            if has_data:
                catalogs_with_data.append(catalog)
        
        # Add catalog information
        combined_info['catalogs'] = catalogs_with_data
        combined_info['found_in_catalogs'] = len(catalogs_with_data) > 0
        
        return combined_info
    
    def _combine_kbo_results(self, 
                           results: Dict[str, Any], 
                           ra: float,
                           dec: float) -> Dict[str, Any]:
        """
        Combine KBO-specific results from multiple catalogs.
        
        Args:
            results: Dictionary with KBO results from each catalog.
            ra: Search center right ascension.
            dec: Search center declination.
            
        Returns:
            Dictionary with combined KBO results.
        """
        combined_kbos = []
        
        # Process each catalog
        for catalog, catalog_results in results.items():
            if catalog == 'combined':
                continue
                
            # Handle errors
            if 'error' in catalog_results:
                logger.warning(f"Error in {catalog} KBO results: {catalog_results['error']}")
                continue
            
            # Extract KBOs based on catalog-specific structure
            kbos = []
            
            if catalog == 'mpc':
                kbos = catalog_results.get('objects', [])
            elif catalog == 'jpl':
                kbos = catalog_results.get('data', [])
            elif catalog == 'skybot':
                kbos = catalog_results.get('tno_candidates', [])
            elif catalog == 'panstarrs':
                kbos = catalog_results.get('data', [])
            elif catalog == 'ossos':
                kbos = catalog_results if isinstance(catalog_results, list) else []
            
            # Convert each KBO to a common format
            for kbo in kbos:
                unified_kbo = self._convert_to_unified_format(catalog, kbo)
                unified_kbo['catalog'] = catalog
                unified_kbo['object_type'] = 'kbo'  # Mark as KBO
                
                # Calculate separation from search center if coords available
                if 'ra' in unified_kbo and 'dec' in unified_kbo:
                    kbo_ra = unified_kbo['ra']
                    kbo_dec = unified_kbo['dec']
                    separation = calculate_separation(ra, dec, kbo_ra, kbo_dec)
                    unified_kbo['separation'] = separation
                
                combined_kbos.append(unified_kbo)
        
        # Sort by separation if available
        combined_kbos.sort(key=lambda x: x.get('separation', float('inf')))
        
        return {
            'search': {
                'ra': ra,
                'dec': dec
            },
            'total_kbos': len(combined_kbos),
            'kbos': combined_kbos
        }
    
    def _find_matches_in_catalog(self, 
                               catalog: str, 
                               catalog_results: Dict[str, Any], 
                               ra: float, 
                               dec: float, 
                               max_separation: float) -> Dict[str, Any]:
        """
        Find objects in catalog results that match a position.
        
        Args:
            catalog: Catalog name.
            catalog_results: Results from the catalog.
            ra: Target right ascension.
            dec: Target declination.
            max_separation: Maximum separation for a match in degrees.
            
        Returns:
            Dictionary with match information.
        """
        matches = []
        
        # Extract objects based on catalog-specific structure
        objects = []
        
        if catalog == 'mpc':
            objects = catalog_results.get('objects', [])
        elif catalog == 'jpl':
            objects = catalog_results.get('data', [])
        elif catalog == 'skybot':
            objects = catalog_results.get('data', [])
        elif catalog == 'panstarrs':
            objects = catalog_results.get('data', [])
        elif catalog == 'ossos':
            objects = catalog_results if isinstance(catalog_results, list) else []
        
        # Check each object for a match
        for obj in objects:
            obj_coords = self._extract_object_coordinates(catalog, obj)
            
            if obj_coords:
                obj_ra, obj_dec = obj_coords
                separation = calculate_separation(ra, dec, obj_ra, obj_dec)
                
                if separation <= max_separation:
                    # Convert to unified format for consistency
                    unified_obj = self._convert_to_unified_format(catalog, obj)
                    unified_obj['separation'] = separation
                    matches.append(unified_obj)
        
        # Sort matches by separation
        matches.sort(key=lambda x: x.get('separation', float('inf')))
        
        return {
            'catalog': catalog,
            'matches': matches,
            'match_count': len(matches),
            'has_match': len(matches) > 0
        }
    
    def _extract_object_coordinates(self, catalog: str, obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        Extract coordinates from an object based on catalog-specific structure.
        
        Args:
            catalog: Catalog name.
            obj: Object data from the catalog.
            
        Returns:
            Tuple of (RA, Dec) in decimal degrees, or None if not available.
        """
        if catalog == 'mpc':
            ra = obj.get('ra')
            dec = obj.get('dec')
            if ra is not None and dec is not None:
                return ra, dec
        elif catalog == 'jpl':
            # JPL structure varies based on query
            if 'ra' in obj and 'dec' in obj:
                return obj['ra'], obj['dec']
        elif catalog == 'skybot':
            ra = obj.get('ra')
            dec = obj.get('dec')
            if ra is not None and dec is not None:
                return float(ra), float(dec)
        elif catalog == 'panstarrs':
            # Try different column names used by Pan-STARRS
            for ra_col in ['ra', 'raMean', 'raStack']:
                for dec_col in ['dec', 'decMean', 'decStack']:
                    if ra_col in obj and dec_col in obj:
                        return obj[ra_col], obj[dec_col]
        elif catalog == 'ossos':
            # OSSOS might have ra/dec directly or as part of a pandas Series
            if hasattr(obj, 'ra') and hasattr(obj, 'dec'):
                return obj.ra, obj.dec
            if isinstance(obj, dict) and 'ra' in obj and 'dec' in obj:
                return obj['ra'], obj['dec']
        
        return None
    
    def _convert_to_unified_format(self, catalog: str, obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an object to a unified format.
        
        Args:
            catalog: Catalog name.
            obj: Object data from the catalog.
            
        Returns:
            Dictionary with object in unified format.
        """
        unified = {
            'catalog': catalog,
            'original_data': obj  # Keep original for reference
        }
        
        # Extract common fields based on catalog
        if catalog == 'mpc':
            self._extract_mpc_fields(obj, unified)
        elif catalog == 'jpl':
            self._extract_jpl_fields(obj, unified)
        elif catalog == 'skybot':
            self._extract_skybot_fields(obj, unified)
        elif catalog == 'panstarrs':
            self._extract_panstarrs_fields(obj, unified)
        elif catalog == 'ossos':
            self._extract_ossos_fields(obj, unified)
        
        return unified
    
    def _extract_mpc_fields(self, obj: Dict[str, Any], unified: Dict[str, Any]) -> None:
        """Extract fields from MPC object to unified format."""
        # Basic identification
        unified['id'] = obj.get('number', obj.get('designation', 'Unknown'))
        unified['name'] = obj.get('name', unified['id'])
        
        # Coordinates
        unified['ra'] = obj.get('ra')
        unified['dec'] = obj.get('dec')
        
        # Orbit elements
        unified['a'] = obj.get('a')  # Semi-major axis
        unified['e'] = obj.get('e')  # Eccentricity
        unified['i'] = obj.get('i')  # Inclination
        
        # Classification
        unified['object_type'] = obj.get('object_type', 'Unknown')
    
    def _extract_jpl_fields(self, obj: Dict[str, Any], unified: Dict[str, Any]) -> None:
        """Extract fields from JPL object to unified format."""
        # Basic identification
        unified['id'] = obj.get('spkid', obj.get('full_name', 'Unknown'))
        unified['name'] = obj.get('name', unified['id'])
        
        # JPL often has nested structure
        orbit = obj.get('orbit', {})
        
        # Coordinates (JPL doesn't always provide these directly)
        if 'ra' in obj and 'dec' in obj:
            unified['ra'] = obj['ra']
            unified['dec'] = obj['dec']
        
        # Orbit elements
        unified['a'] = orbit.get('a')  # Semi-major axis
        unified['e'] = orbit.get('e')  # Eccentricity
        unified['i'] = orbit.get('i')  # Inclination
        
        # Classification
        unified['object_type'] = obj.get('kind', 'Unknown')
    
    def _extract_skybot_fields(self, obj: Dict[str, Any], unified: Dict[str, Any]) -> None:
        """Extract fields from SkyBoT object to unified format."""
        # Basic identification
        unified['id'] = obj.get('number', obj.get('name', 'Unknown'))
        unified['name'] = obj.get('name', unified['id'])
        
        # Coordinates
        if 'ra' in obj and 'dec' in obj:
            try:
                unified['ra'] = float(obj['ra'])
                unified['dec'] = float(obj['dec'])
            except (ValueError, TypeError):
                pass
        
        # Motion
        if 'dra' in obj and 'ddec' in obj:
            try:
                unified['dra'] = float(obj['dra'])  # RA motion
                unified['ddec'] = float(obj['ddec'])  # Dec motion
            except (ValueError, TypeError):
                pass
        
        # Classification
        unified['object_type'] = obj.get('type', 'Unknown')
    
    def _extract_panstarrs_fields(self, obj: Dict[str, Any], unified: Dict[str, Any]) -> None:
        """Extract fields from Pan-STARRS object to unified format."""
        # Basic identification
        unified['id'] = obj.get('objID', 'Unknown')
        
        # Coordinates - try different column names
        for ra_col in ['ra', 'raMean', 'raStack']:
            if ra_col in obj:
                unified['ra'] = obj[ra_col]
                break
                
        for dec_col in ['dec', 'decMean', 'decStack']:
            if dec_col in obj:
                unified['dec'] = obj[dec_col]
                break
        
        # Magnitudes
        for band in ['g', 'r', 'i', 'z', 'y']:
            mag_col = f'{band}MeanPSFMag'
            if mag_col in obj:
                unified[f'{band}_mag'] = obj[mag_col]
        
        # If this is a moving object candidate, include that info
        if 'moving_object_candidate' in obj:
            unified['moving_object_candidate'] = obj['moving_object_candidate']
            if 'moving_object_reasons' in obj:
                unified['moving_object_reasons'] = obj['moving_object_reasons']
    
    def _extract_ossos_fields(self, obj: Dict[str, Any], unified: Dict[str, Any]) -> None:
        """Extract fields from OSSOS object to unified format."""
        # Handle both dict and pandas Series
        if hasattr(obj, 'to_dict'):
            obj_dict = obj.to_dict()
        else:
            obj_dict = obj
            
        # Basic identification
        for id_field in ['name', 'id', 'designation']:
            if id_field in obj_dict:
                unified['id'] = obj_dict[id_field]
                unified['name'] = obj_dict.get('name', unified['id'])
                break
        
        # Coordinates
        if 'ra' in obj_dict and 'dec' in obj_dict:
            unified['ra'] = obj_dict['ra']
            unified['dec'] = obj_dict['dec']
        
        # Orbit elements
        for elem in ['a', 'e', 'i']:
            if elem in obj_dict:
                unified[elem] = obj_dict[elem]
        
        # Classification
        if 'dynamical_class' in obj_dict:
            unified['object_type'] = obj_dict['dynamical_class']
    
    def _extract_mpc_object_properties(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key properties from MPC object."""
        properties = {}
        
        # Basic info
        properties['designation'] = obj.get('designation')
        properties['name'] = obj.get('name')
        properties['object_type'] = obj.get('object_type')
        
        # Orbit elements if available
        if 'orbit' in obj:
            orbit = obj['orbit']
            properties['a'] = orbit.get('a')
            properties['e'] = orbit.get('e')
            properties['i'] = orbit.get('i')
            
            # Compute perihelion and aphelion if possible
            if 'a' in orbit and 'e' in orbit:
                a = orbit['a']
                e = orbit['e']
                properties['perihelion'] = a * (1 - e)
                properties['aphelion'] = a * (1 + e)
        
        return properties
    
    def _extract_jpl_object_properties(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key properties from JPL object."""
        properties = {}
        
        # Find the main data object
        data = obj.get('data', [{}])[0]
        
        # Basic info
        properties['designation'] = data.get('full_name')
        properties['name'] = data.get('name')
        properties['object_type'] = data.get('kind')
        
        # Orbit elements
        orbit = data.get('orbit', {})
        properties['a'] = orbit.get('a')
        properties['e'] = orbit.get('e')
        properties['i'] = orbit.get('i')
        
        # Compute perihelion and aphelion if possible
        if 'a' in orbit and 'e' in orbit:
            a = orbit['a']
            e = orbit['e']
            properties['perihelion'] = a * (1 - e)
            properties['aphelion'] = a * (1 + e)
        
        return properties
    
    def _extract_ossos_object_properties(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key properties from OSSOS object."""
        properties = {}
        
        # Handle both dict and pandas Series
        if hasattr(obj, 'to_dict'):
            obj_dict = obj.to_dict()
        else:
            obj_dict = obj
        
        # Basic info
        for field in ['name', 'designation', 'id']:
            if field in obj_dict:
                properties['designation'] = obj_dict[field]
                break
                
        properties['object_type'] = obj_dict.get('dynamical_class', 'KBO')
        
        # Orbit elements
        for elem in ['a', 'e', 'i']:
            if elem in obj_dict:
                properties[elem] = obj_dict[elem]
        
        # Compute perihelion and aphelion if possible
        if 'a' in obj_dict and 'e' in obj_dict:
            a = obj_dict['a']
            e = obj_dict['e']
            properties['perihelion'] = a * (1 - e)
            properties['aphelion'] = a * (1 + e)
        
        return properties

# Helper function for direct candidate lookup
def lookup_candidate(candidate, catalogs=None):
    """
    Look up a candidate in multiple catalogs.
    
    Args:
        candidate: Dictionary with candidate data (must include 'ra' and 'dec').
        catalogs: List of catalogs to query (default: all available).
        
    Returns:
        Dictionary with lookup results.
    """
    # Create a temporary QueryManager
    query_manager = QueryManager()
    
    # Extract position
    ra = candidate.get('ra')
    dec = candidate.get('dec')
    
    if ra is None or dec is None:
        raise ValueError("Candidate must have 'ra' and 'dec' fields")
    
    # Extract epoch if available
    epoch = candidate.get('epoch')
    
    # Perform search
    search_radius = 0.1  # degrees
    return query_manager.search_by_coordinates(
        ra=ra, dec=dec, radius=search_radius, epoch=epoch, catalogs=catalogs)

# Helper function for batch candidate lookup
def lookup_candidates(candidates, catalogs=None, parallel=True):
    """
    Look up multiple candidates in catalogs.
    
    Args:
        candidates: List of dictionaries with candidate data.
        catalogs: List of catalogs to query (default: all available).
        parallel: Whether to process candidates in parallel.
        
    Returns:
        List of dictionaries with lookup results.
    """
    # Create a QueryManager with parallel querying if requested
    query_manager = QueryManager(parallel_queries=parallel)
    
    results = []
    
    # Process each candidate
    for candidate in candidates:
        # Extract position
        ra = candidate.get('ra')
        dec = candidate.get('dec')
        
        if ra is None or dec is None:
            results.append({'error': "Candidate must have 'ra' and 'dec' fields"})
            continue
        
        # Extract epoch if available
        epoch = candidate.get('epoch')
        
        # Perform search
        search_radius = 0.1  # degrees
        result = query_manager.search_by_coordinates(
            ra=ra, dec=dec, radius=search_radius, epoch=epoch, catalogs=catalogs)
        
        results.append(result)
    
    return results