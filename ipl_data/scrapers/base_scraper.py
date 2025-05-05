"""
Base scraper class for IPL data.
"""

import logging
import random
import time
from pathlib import Path
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import backoff
from bs4 import BeautifulSoup
import json
import os
import cloudscraper

class BaseScraper:
    """Base class for all scrapers."""
    
    def __init__(self, cache_dir: str = "cache", data_dir: str = "ipl_data/data/raw"):
        """Initialize base scraper."""
        self.cache_dir = Path(cache_dir)
        self.data_dir = Path(data_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # Set up cloudscraper session
        self.session = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
        
        # Set up retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set up user agents
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        ]
        
        # Set up headers
        self.headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "TE": "trailers",
            "DNT": "1"
        }
    
    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for URL."""
        return self.cache_dir / f"{hash(url)}.html"
    
    def _load_from_cache(self, url: str) -> Optional[str]:
        """Load content from cache if available."""
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            self.logger.info(f"Loading from cache: {url}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        return None
    
    def _save_to_cache(self, url: str, content: str):
        """Save content to cache."""
        cache_path = self._get_cache_path(url)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.HTTPError),
        max_tries=3
    )
    def get(self, url: str, use_cache: bool = True) -> BeautifulSoup:
        """
        Get HTML content from URL with caching and retries.
        
        Args:
            url: URL to fetch
            use_cache: Whether to use cache
            
        Returns:
            BeautifulSoup object
        """
        if use_cache:
            cached_content = self._load_from_cache(url)
            if cached_content:
                return BeautifulSoup(cached_content, "html.parser")
        
        # Add random delay
        time.sleep(random.uniform(2, 5))
        
        # Set random user agent
        self.headers["User-Agent"] = random.choice(self.user_agents)
        
        try:
            # Add referer header
            self.headers["Referer"] = "https://www.google.com/"
            
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=30,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Save to cache
            if use_cache:
                self._save_to_cache(url, response.text)
            
            return BeautifulSoup(response.text, "html.parser")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            raise
    
    def _parse_json(self, text: str) -> dict:
        """Parse JSON text safely."""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            return {}
    
    def _save_json(self, data: dict, filename: str):
        """Save data as JSON file."""
        output_path = self.cache_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def _load_json(self, filename: str) -> dict:
        """Load data from JSON file."""
        input_path = self.cache_dir / filename
        if input_path.exists():
            with open(input_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def close(self):
        """Close the session and cleanup resources."""
        if hasattr(self, 'session'):
            self.session.close() 