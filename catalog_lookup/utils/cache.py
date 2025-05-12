"""
cache.py - Caching utilities for API queries

This module provides utilities to cache API responses, avoiding redundant queries
and improving performance for the catalog lookup system.
"""

import os
import json
import hashlib
import pickle
import time
import logging
from typing import Dict, Any, Optional, Union, Callable
from functools import wraps

# Set up logging
logger = logging.getLogger(__name__)

class QueryCache:
    """
    Cache for API query results.
    
    This class provides a disk-based cache for API responses to avoid
    redundant queries and improve performance.
    """
    
    def __init__(self, cache_dir: str = None, expiration: int = 86400):
        """
        Initialize the query cache.
        
        Args:
            cache_dir: Directory to store cache files. If None, uses "./cache".
            expiration: Cache expiration time in seconds (default: 24 hours).
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cache")
        self.expiration = expiration
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key (e.g., query string).
            
        Returns:
            Cached value if present and not expired, None otherwise.
        """
        cache_file = self._get_cache_path(key)
        
        if not os.path.exists(cache_file):
            self.misses += 1
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if cache is expired
            if time.time() - cached_data.get('timestamp', 0) > self.expiration:
                logger.debug(f"Cache expired for key: {key}")
                self.misses += 1
                return None
            
            logger.debug(f"Cache hit for key: {key}")
            self.hits += 1
            return cached_data.get('data')
            
        except (pickle.PickleError, EOFError, IOError) as e:
            logger.warning(f"Error reading cache file: {e}")
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key (e.g., query string).
            value: Value to cache.
        """
        cache_file = self._get_cache_path(key)
        
        try:
            cached_data = {
                'timestamp': time.time(),
                'data': value
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
                
            logger.debug(f"Cached value for key: {key}")
            
        except (pickle.PickleError, IOError) as e:
            logger.warning(f"Error writing to cache file: {e}")
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cached value.
        
        Args:
            key: Cache key to invalidate.
            
        Returns:
            True if the cache entry was invalidated, False otherwise.
        """
        cache_file = self._get_cache_path(key)
        
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                logger.debug(f"Invalidated cache for key: {key}")
                return True
            except IOError as e:
                logger.warning(f"Error removing cache file: {e}")
                return False
        
        return False
    
    def clear(self) -> int:
        """
        Clear all cached values.
        
        Returns:
            Number of cache entries cleared.
        """
        count = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.startswith('cache_'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                    count += 1
                except IOError as e:
                    logger.warning(f"Error removing cache file {filename}: {e}")
        
        logger.info(f"Cleared {count} cache entries")
        return count
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics.
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total': total,
            'hit_rate': hit_rate
        }
    
    def _get_cache_path(self, key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            key: Cache key.
            
        Returns:
            File path for the cache entry.
        """
        # Hash the key to get a filename
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"cache_{key_hash}.pkl")


def cached(cache: QueryCache, key_fn: Callable = None):
    """
    Decorator to cache function results.
    
    Args:
        cache: QueryCache instance to use.
        key_fn: Optional function to generate a cache key from function arguments.
               If None, uses repr(args) + repr(kwargs).
    
    Returns:
        Decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__name__}:{repr(args)}:{repr(kwargs)}"
            
            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Cache the result
            cache.set(key, result)
            
            return result
        return wrapper
    return decorator


# Create a global cache instance
default_cache = QueryCache()