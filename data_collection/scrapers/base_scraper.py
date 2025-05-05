"""
Base scraper class for IPL data collection.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import random
import time
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)

class BaseScraper(ABC):
    """Base class for all IPL data scrapers."""
    
    def __init__(self, proxy_list: Optional[List[str]] = None):
        """Initialize the scraper with optional proxy list."""
        self.proxy_list = proxy_list or []
        self.user_agent = UserAgent()
        self.driver = None
        
    def setup_driver(self) -> None:
        """Set up the Selenium WebDriver with proxy and user agent."""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            
            # Set random user agent
            chrome_options.add_argument(f'user-agent={self.user_agent.random}')
            
            # Set proxy if available
            if self.proxy_list:
                proxy = random.choice(self.proxy_list)
                chrome_options.add_argument(f'--proxy-server={proxy}')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)
            
        except Exception as e:
            logger.error(f"Error setting up WebDriver: {str(e)}")
            raise
            
    def wait_for_element(self, by: By, value: str, timeout: int = 10) -> None:
        """Wait for an element to be present on the page."""
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
        except TimeoutException:
            logger.warning(f"Timeout waiting for element: {value}")
            raise
            
    def safe_get(self, url: str, retries: int = 3) -> None:
        """Safely get a URL with retry mechanism."""
        for attempt in range(retries):
            try:
                self.driver.get(url)
                time.sleep(random.uniform(2, 5))  # Random delay
                return
            except WebDriverException as e:
                if attempt == retries - 1:
                    logger.error(f"Failed to get URL after {retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(5)  # Wait before retry
                
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.error(f"Error cleaning up WebDriver: {str(e)}")
                
    @abstractmethod
    def scrape_match_data(self, match_url: str) -> Dict:
        """Scrape data for a specific match."""
        pass
        
    @abstractmethod
    def scrape_player_data(self, player_url: str) -> Dict:
        """Scrape data for a specific player."""
        pass
        
    @abstractmethod
    def scrape_team_data(self, team_url: str) -> Dict:
        """Scrape data for a specific team."""
        pass 