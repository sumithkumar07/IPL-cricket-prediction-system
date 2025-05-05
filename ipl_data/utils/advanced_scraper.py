"""
Advanced base scraper with Selenium, proxy rotation, and rate limiting.
"""

import logging
import logging.handlers
import time
import random
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from urllib.robotparser import RobotFileParser
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import hashlib
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    SELENIUM_CONFIG,
    PROXY_CONFIG,
    RATE_LIMIT_CONFIG,
    ROBOTS_CONFIG,
    BROWSER_PROFILES,
    CACHE_CONFIG,
    LOG_CONFIG,
)


class AdvancedScraper:
    """Advanced base scraper with Selenium, proxy rotation, and rate limiting."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.ua = UserAgent()
        self.driver = None
        self.setup_logging()
        self.setup_robots_parser()
        self.setup_cache()
        self.setup_rate_limiting()
        self.setup_proxy_rotation()
        self.setup_selenium()

    def setup_logging(self):
        """Configure advanced logging with rotation."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / LOG_CONFIG["log_file"]

        # Add a console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LOG_CONFIG["log_format"]))

        # Add a file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOG_CONFIG["max_log_size"],
            backupCount=LOG_CONFIG["backup_count"],
        )
        file_handler.setFormatter(logging.Formatter(LOG_CONFIG["log_format"]))

        # Configure logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, LOG_CONFIG["log_level"]))
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # Prevent log messages from propagating to the root logger
        self.logger.propagate = False

    def setup_robots_parser(self):
        """Setup robots.txt parser."""
        if ROBOTS_CONFIG["respect_robots_txt"]:
            self.robots_parser = RobotFileParser()
            self.robots_parser.set_url(f"{self.base_url}/robots.txt")
            try:
                self.robots_parser.read()
            except Exception as e:
                self.logger.warning(f"Could not read robots.txt: {str(e)}")
                self.robots_parser = None

    def setup_cache(self):
        """Setup caching system."""
        if CACHE_CONFIG["enabled"]:
            self.cache_dir = Path(CACHE_CONFIG["cache_dir"])
            self.cache_dir.mkdir(exist_ok=True)

    def setup_rate_limiting(self):
        """Setup rate limiting."""
        self.request_times = []
        self.rate_limit_lock = threading.Lock()
        self.request_queue = Queue()
        self.worker_pool = ThreadPoolExecutor(
            max_workers=RATE_LIMIT_CONFIG["concurrent_requests"]
        )

    def setup_proxy_rotation(self):
        """Setup proxy rotation."""
        if PROXY_CONFIG["enabled"]:
            self.proxy_list = PROXY_CONFIG["proxy_list"]
            self.current_proxy = None
            self.proxy_fails = 0
            self.requests_since_rotation = 0
            self.rotate_proxy()

    def setup_selenium(self):
        """Setup Selenium WebDriver."""
        try:
            options = Options()
            if SELENIUM_CONFIG["headless"]:
                options.add_argument("--headless=new")

            # Add browser profile settings
            for arg in BROWSER_PROFILES["chrome"]["arguments"]:
                options.add_argument(arg)

            for pref, value in BROWSER_PROFILES["chrome"]["preferences"].items():
                options.add_experimental_option("prefs", {pref: value})

            # Set up custom user agent
            options.add_argument(f'user-agent={SELENIUM_CONFIG["user_agent"]}')

            try:
                # Try using webdriver_manager first
                self.logger.info(
                    "Attempting to install ChromeDriver using webdriver_manager..."
                )
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
                self.logger.info(
                    "ChromeDriver installed successfully using webdriver_manager"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to install ChromeDriver using webdriver_manager: {str(e)}"
                )

                # Fallback to manual path if specified
                if SELENIUM_CONFIG["driver_path"]:
                    self.logger.info(
                        f"Using specified ChromeDriver path: {SELENIUM_CONFIG['driver_path']}"
                    )
                    service = Service(executable_path=SELENIUM_CONFIG["driver_path"])
                    self.driver = webdriver.Chrome(service=service, options=options)
                else:
                    # Try default Chrome installation
                    self.logger.info("Attempting to use default Chrome installation...")
                    self.driver = webdriver.Chrome(options=options)

            if self.driver:
                self.driver.set_page_load_timeout(SELENIUM_CONFIG["page_load_timeout"])
                self.driver.implicitly_wait(SELENIUM_CONFIG["implicit_wait"])
                self.driver.set_window_size(*SELENIUM_CONFIG["window_size"])
                self.logger.info("Selenium WebDriver setup completed successfully")

        except Exception as e:
            self.logger.error(f"Failed to setup Selenium: {str(e)}", exc_info=True)
            self.driver = None

    def rotate_proxy(self):
        """Rotate to a new proxy."""
        if not PROXY_CONFIG["enabled"] or not self.proxy_list:
            return

        self.current_proxy = random.choice(self.proxy_list)
        self.proxy_fails = 0
        self.requests_since_rotation = 0
        self.session.proxies = {"http": self.current_proxy, "https": self.current_proxy}
        self.logger.info(f"Rotated to new proxy: {self.current_proxy}")

    def get_cache_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for URL and parameters."""
        key = f"{url}{json.dumps(params) if params else ''}"
        return hashlib.md5(key.encode()).hexdigest()

    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if available and not expired."""
        if not CACHE_CONFIG["enabled"]:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    if datetime.now() - cache_data["timestamp"] < timedelta(
                        seconds=CACHE_CONFIG["cache_expiry"]
                    ):
                        return cache_data["data"]
            except Exception as e:
                self.logger.warning(f"Cache read error: {str(e)}")
        return None

    def save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache."""
        if not CACHE_CONFIG["enabled"]:
            return

        try:
            cache_data = {"timestamp": datetime.now(), "data": data}
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            self.logger.warning(f"Cache write error: {str(e)}")

    def check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        if not ROBOTS_CONFIG["respect_robots_txt"] or not self.robots_parser:
            return True

        try:
            return self.robots_parser.can_fetch(ROBOTS_CONFIG["user_agent"], url)
        except Exception as e:
            self.logger.warning(f"Robots.txt check failed: {str(e)}")
            return True

    def respect_rate_limit(self):
        """Ensure we respect rate limits."""
        with self.rate_limit_lock:
            now = time.time()
            # Remove old request times
            self.request_times = [t for t in self.request_times if now - t < 60]

            if len(self.request_times) >= RATE_LIMIT_CONFIG["requests_per_minute"]:
                sleep_time = 60 - (now - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.request_times.append(now)

    def make_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        use_selenium: bool = False,
    ) -> Optional[Any]:
        """Make an HTTP request with all improvements."""
        if not self.check_robots_txt(url):
            self.logger.warning(f"URL not allowed by robots.txt: {url}")
            return None

        cache_key = self.get_cache_key(url, params)
        cached_data = self.get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        self.respect_rate_limit()

        if use_selenium and self.driver:
            try:
                self.driver.get(url)
                WebDriverWait(self.driver, SELENIUM_CONFIG["page_load_timeout"]).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                return self.driver.page_source
            except Exception as e:
                self.logger.error(f"Selenium request failed: {str(e)}")
                return None

        for attempt in range(RATE_LIMIT_CONFIG["max_retries"]):
            try:
                headers = {
                    "User-Agent": self.ua.random,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Connection": "keep-alive",
                }

                response = self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    timeout=SELENIUM_CONFIG["page_load_timeout"],
                )
                response.raise_for_status()

                if PROXY_CONFIG["enabled"]:
                    self.requests_since_rotation += 1
                    if (
                        self.requests_since_rotation
                        >= PROXY_CONFIG["proxy_rotation_interval"]
                    ):
                        self.rotate_proxy()

                data = response.text
                self.save_to_cache(cache_key, data)
                return data

            except Exception as e:
                self.logger.error(
                    f"Request failed (attempt {attempt + 1}/{RATE_LIMIT_CONFIG['max_retries']}): {str(e)}"
                )

                if PROXY_CONFIG["enabled"]:
                    self.proxy_fails += 1
                    if self.proxy_fails >= PROXY_CONFIG["max_fails"]:
                        self.rotate_proxy()

                if attempt == RATE_LIMIT_CONFIG["max_retries"] - 1:
                    return None

                time.sleep(
                    RATE_LIMIT_CONFIG["retry_delay"]
                    * (RATE_LIMIT_CONFIG["backoff_factor"] ** attempt)
                )

        return None

    def get_soup(
        self, url: str, params: Optional[Dict] = None, use_selenium: bool = False
    ) -> Optional[BeautifulSoup]:
        """Get BeautifulSoup object from URL."""
        content = self.make_request(url, params=params, use_selenium=use_selenium)
        if content:
            return BeautifulSoup(content, "lxml")
        return None

    def save_data(self, data: Any, filename: str, format: str = "csv"):
        """Save scraped data to file."""
        output_dir = Path("data/raw")
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

    def cleanup(self):
        """Cleanup resources."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                self.logger.error(f"Error closing Selenium driver: {str(e)}")

        if hasattr(self, "worker_pool"):
            self.worker_pool.shutdown()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
