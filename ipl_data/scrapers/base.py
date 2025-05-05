"""
Base scraper class with advanced anti-detection measures.
"""

import logging
import random
import time
from pathlib import Path
from typing import Optional, Dict, Any
import json
from urllib.parse import urlparse

import cloudscraper
from fake_useragent import UserAgent
import backoff
import requests
from bs4 import BeautifulSoup

class BaseScraper:
    def __init__(self, cache_dir: Optional[str] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize user agent rotator with more realistic agents
        self.ua = UserAgent(browsers=['chrome', 'firefox', 'edge'])
        
        # Initialize session with enhanced settings
        self.session = self._create_session()
        
        # Configure backoff and rate limiting
        self.max_retries = 5
        self.min_delay = 3
        self.max_delay = 15
        
        # Track request history for rate limiting
        self._last_request_time = 0
        self._request_count = 0
        self._max_requests_per_minute = 20
    
    def _create_session(self) -> cloudscraper.CloudScraper:
        """Create a new session with enhanced anti-detection measures."""
        session = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False,
                'custom': random.choice([
                    'Chrome/122.0.0.0',
                    'Chrome/121.0.0.0',
                    'Firefox/123.0',
                    'Edge/122.0.0.0'
                ])
            },
            delay=random.uniform(1, 3)
        )
        
        # Set up headers with more realistic values
        session.headers.update({
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'TE': 'trailers'
        })
        return session
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_diff = current_time - self._last_request_time
        
        # Reset counter if more than a minute has passed
        if time_diff > 60:
            self._request_count = 0
            self._last_request_time = current_time
        
        # If we've made too many requests in the last minute, wait
        if self._request_count >= self._max_requests_per_minute:
            sleep_time = 60 - time_diff
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, waiting {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self._request_count = 0
                self._last_request_time = time.time()
    
    def _get_referer(self, url: str) -> str:
        """Generate a realistic referer URL."""
        parsed = urlparse(url)
        referers = [
            f"https://www.google.com/search?q={parsed.netloc}",
            f"https://www.bing.com/search?q={parsed.netloc}",
            f"https://duckduckgo.com/?q={parsed.netloc}",
            f"https://{parsed.netloc}"
        ]
        return random.choice(referers)
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, Exception),
        max_tries=5,
        max_time=300
    )
    def _make_request(self, url: str, params: Optional[Dict[Any, Any]] = None) -> BeautifulSoup:
        """Make a request with enhanced retry logic and anti-detection measures."""
        self._enforce_rate_limit()
        
        # Add random delay between requests
        time.sleep(random.uniform(self.min_delay, self.max_delay))
        
        # Update session with new headers
        self.session.headers.update({
            'User-Agent': self.ua.random,
            'Referer': self._get_referer(url)
        })
        
        try:
            # Check cache first
            cache_key = f"{hash(url)}-{hash(str(params))}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return BeautifulSoup(cached_data, 'lxml')
            
            # Make request with timeout
            response = self.session.get(
                url,
                params=params,
                timeout=(10, 30),  # (connect timeout, read timeout)
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Update request tracking
            self._request_count += 1
            self._last_request_time = time.time()
            
            # Cache the response
            self._save_to_cache(cache_key, response.text)
            
            # Log successful request
            self.logger.info(f"Successfully fetched {url}")
            
            return BeautifulSoup(response.text, 'lxml')
        
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            raise
    
    def _get_cached_data(self, cache_key: str) -> Optional[str]:
        """Get data from cache if available and not expired."""
        cache_file = self.cache_dir / f"{cache_key}.html"
        if cache_file.exists():
            # Check if cache is less than 30 minutes old
            if time.time() - cache_file.stat().st_mtime < 1800:
                self.logger.info(f"Using cached data for {cache_key}")
                return cache_file.read_text(encoding='utf-8')
            else:
                cache_file.unlink()  # Delete expired cache
        return None
    
    def _save_to_cache(self, cache_key: str, content: str) -> None:
        """Save data to cache with error handling."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.html"
            cache_file.write_text(content, encoding='utf-8')
            self.logger.info(f"Saved data to cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def close(self) -> None:
        """Close the session and cleanup resources."""
        if self.session:
            self.session.close()
            self.logger.info("Closed scraper session") 