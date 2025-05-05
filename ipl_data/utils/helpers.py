"""
Helper utilities for IPL data scraping.
"""

import logging
import time
import requests
from typing import Optional, Dict, Any
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SCRAPING_CONFIG, LOGGING_CONFIG, EXPORT_CONFIG


class BaseScraper:
    """Base class for all scrapers with common functionality."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.ua = UserAgent()
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the scraper."""
        logging.basicConfig(
            filename=LOGGING_CONFIG["log_file"],
            level=getattr(logging, LOGGING_CONFIG["log_level"]),
            format=LOGGING_CONFIG["log_format"],
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_headers(self) -> Dict[str, str]:
        """Generate headers for HTTP requests."""
        return {
            "User-Agent": self.ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

    def make_request(
        self, url: str, method: str = "GET", params: Optional[Dict] = None
    ) -> Optional[requests.Response]:
        """Make an HTTP request with retry logic and error handling."""
        for attempt in range(SCRAPING_CONFIG["max_retries"]):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=self.get_headers(),
                    params=params,
                    timeout=SCRAPING_CONFIG["timeout"],
                )
                response.raise_for_status()
                time.sleep(SCRAPING_CONFIG["request_delay"])
                return response
            except requests.exceptions.RequestException as e:
                self.logger.error(
                    f"Request failed (attempt {attempt + 1}/{SCRAPING_CONFIG['max_retries']}): {str(e)}"
                )
                if attempt == SCRAPING_CONFIG["max_retries"] - 1:
                    self.logger.error(f"Max retries reached for URL: {url}")
                    return None
                time.sleep(SCRAPING_CONFIG["request_delay"] * (attempt + 1))
        return None

    def get_soup(
        self, url: str, params: Optional[Dict] = None
    ) -> Optional[BeautifulSoup]:
        """Get BeautifulSoup object from URL."""
        response = self.make_request(url, params=params)
        if response:
            return BeautifulSoup(response.content, "lxml")
        return None

    def save_data(self, data: Any, filename: str, format: str = "csv"):
        """Save scraped data to file."""
        output_dir = Path(EXPORT_CONFIG["raw_data_path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / f"{filename}.{format}"
        try:
            if format == "csv":
                if isinstance(data, pd.DataFrame):
                    data.to_csv(filepath, index=False)
                else:
                    pd.DataFrame(data).to_csv(filepath, index=False)
            elif format == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Data saved successfully to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving data to {filepath}: {str(e)}")

    def validate_data(self, data: Any) -> bool:
        """Validate scraped data."""
        if data is None:
            return False
        if isinstance(data, (list, dict)):
            return len(data) > 0
        if isinstance(data, pd.DataFrame):
            return not data.empty
        return True
