"""
rate_limiter.py - Rate limiting utilities for API queries

This module provides utilities to limit the rate of API calls, preventing
abuse of external services and conforming to their rate limits.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import datetime

# Set up logging
logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiter for API calls.
    
    This class implements a token bucket algorithm to limit the rate of API calls.
    """
    
    def __init__(self, 
                 calls_per_second: float = 1.0, 
                 burst: int = 5, 
                 retry_after: int = 1):
        """
        Initialize the rate limiter.
        
        Args:
            calls_per_second: Maximum calls per second.
            burst: Maximum burst size (number of consecutive calls allowed).
            retry_after: Seconds to wait before retrying if rate limited.
        """
        self.calls_per_second = calls_per_second
        self.burst = burst
        self.retry_after = retry_after
        
        # Initialize token bucket
        self.tokens = burst
        self.last_refill = time.time()
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def __call__(self, func):
        """
        Decorator to rate limit a function.
        
        Args:
            func: Function to rate limit.
            
        Returns:
            Rate-limited function.
        """
        def wrapper(*args, **kwargs):
            self.acquire()
            try:
                return func(*args, **kwargs)
            finally:
                pass  # Token was already consumed during acquire
        
        return wrapper
    
    def acquire(self) -> None:
        """
        Acquire a token from the bucket.
        
        This method blocks until a token is available.
        """
        while True:
            with self.lock:
                # Refill tokens based on elapsed time
                now = time.time()
                elapsed = now - self.last_refill
                self.last_refill = now
                
                # Calculate new tokens
                new_tokens = elapsed * self.calls_per_second
                self.tokens = min(self.burst, self.tokens + new_tokens)
                
                # Check if we can acquire a token
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            
            # No tokens available, wait and retry
            wait_time = self.retry_after
            logger.debug(f"Rate limit reached, waiting {wait_time} seconds")
            time.sleep(wait_time)
    
    def wait_if_needed(self) -> None:
        """
        Wait if necessary to satisfy rate limit.
        
        Unlike acquire(), this method doesn't block if tokens are available.
        """
        with self.lock:
            # Refill tokens based on elapsed time
            now = time.time()
            elapsed = now - self.last_refill
            self.last_refill = now
            
            # Calculate new tokens
            new_tokens = elapsed * self.calls_per_second
            self.tokens = min(self.burst, self.tokens + new_tokens)
            
            # Check if we can acquire a token
            if self.tokens >= 1:
                self.tokens -= 1
                return
        
        # No tokens available, wait
        wait_time = 1.0 / self.calls_per_second
        logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
        time.sleep(wait_time)
        
        # Now acquire a token (should be available after waiting)
        self.acquire()


class WindowRateLimiter:
    """
    Sliding window rate limiter for API calls.
    
    This class implements a sliding window algorithm to limit the rate of API calls,
    which is more precise than the token bucket algorithm.
    """
    
    def __init__(self, 
                 max_calls: int = 60, 
                 window_seconds: int = 60, 
                 retry_after: int = 5):
        """
        Initialize the sliding window rate limiter.
        
        Args:
            max_calls: Maximum number of calls in the window.
            window_seconds: Window size in seconds.
            retry_after: Seconds to wait before retrying if rate limited.
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        
        # Queue to track call timestamps
        self.call_times = deque()
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def __call__(self, func):
        """
        Decorator to rate limit a function.
        
        Args:
            func: Function to rate limit.
            
        Returns:
            Rate-limited function.
        """
        def wrapper(*args, **kwargs):
            self.acquire()
            try:
                return func(*args, **kwargs)
            finally:
                pass  # Call was already logged during acquire
        
        return wrapper
    
    def acquire(self) -> None:
        """
        Acquire permission to make a call.
        
        This method blocks until a call is allowed.
        """
        while True:
            with self.lock:
                now = time.time()
                
                # Remove old calls from the window
                while self.call_times and self.call_times[0] < now - self.window_seconds:
                    self.call_times.popleft()
                
                # Check if we can make a call
                if len(self.call_times) < self.max_calls:
                    self.call_times.append(now)
                    return
            
            # Rate limit exceeded, wait and retry
            wait_time = self.retry_after
            logger.debug(f"Rate limit reached, waiting {wait_time} seconds")
            time.sleep(wait_time)
    
    def wait_if_needed(self) -> None:
        """
        Wait if necessary to satisfy rate limit.
        
        Unlike acquire(), this method calculates the exact wait time.
        """
        with self.lock:
            now = time.time()
            
            # Remove old calls from the window
            while self.call_times and self.call_times[0] < now - self.window_seconds:
                self.call_times.popleft()
            
            # Check if we can make a call
            if len(self.call_times) < self.max_calls:
                self.call_times.append(now)
                return
            
            # Calculate wait time until the oldest call expires
            wait_time = self.call_times[0] + self.window_seconds - now
        
        # Wait until we can make a call
        logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
        time.sleep(wait_time)
        
        # Add the call to the window
        with self.lock:
            self.call_times.append(time.time())


# Create common rate limiters for different services
mpc_rate_limiter = RateLimiter(calls_per_second=0.5, burst=2)  # 1 call every 2 seconds
jpl_rate_limiter = RateLimiter(calls_per_second=1.0, burst=5)  # 1 call per second, 5 burst
skybot_rate_limiter = RateLimiter(calls_per_second=0.1, burst=1)  # 1 call every 10 seconds
panstarrs_rate_limiter = WindowRateLimiter(max_calls=60, window_seconds=60)  # 60 calls per minute