"""
Data collection manager for orchestrating the data collection pipeline.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .scrapers.espncricinfo import ESPNCricinfoScraper
from .processors.data_processor import DataProcessor
from .storage.storage_manager import StorageManager

logger = logging.getLogger(__name__)

class DataCollectionManager:
    """Class for managing the data collection pipeline."""
    
    def __init__(self):
        """Initialize the data collection manager."""
        self.scraper = ESPNCricinfoScraper()
        self.processor = DataProcessor()
        self.storage = StorageManager()
        
    def collect_match_data(self, match_url: str) -> Dict:
        """Collect and store data for a specific match."""
        try:
            # Scrape match data
            raw_match_data = self.scraper.scrape_match_data(match_url)
            if not raw_match_data:
                logger.error(f"Failed to scrape match data from {match_url}")
                return None
                
            # Process match data
            processed_match_data = self.processor.process_match_data(raw_match_data)
            if not processed_match_data:
                logger.error("Failed to process match data")
                return None
                
            # Store match data
            match = self.storage.store_match_data(processed_match_data)
            
            # Collect and store player performances
            for performance in raw_match_data.get('player_performances', []):
                processed_performance = self.processor.process_player_performance(performance)
                if processed_performance:
                    self.storage.store_player_performance(processed_performance)
                    
            return processed_match_data
            
        except Exception as e:
            logger.error(f"Error collecting match data: {str(e)}")
            return None
            
    def collect_player_data(self, player_url: str) -> Dict:
        """Collect and store data for a specific player."""
        try:
            # Scrape player data
            raw_player_data = self.scraper.scrape_player_data(player_url)
            if not raw_player_data:
                logger.error(f"Failed to scrape player data from {player_url}")
                return None
                
            # Process player data
            processed_player_data = self.processor.process_player_data(raw_player_data)
            if not processed_player_data:
                logger.error("Failed to process player data")
                return None
                
            # Store player data
            player = self.storage.store_player_data(processed_player_data)
            return processed_player_data
            
        except Exception as e:
            logger.error(f"Error collecting player data: {str(e)}")
            return None
            
    def collect_team_data(self, team_url: str) -> Dict:
        """Collect and store data for a specific team."""
        try:
            # Scrape team data
            raw_team_data = self.scraper.scrape_team_data(team_url)
            if not raw_team_data:
                logger.error(f"Failed to scrape team data from {team_url}")
                return None
                
            # Process team data
            processed_team_data = self.processor.process_team_data(raw_team_data)
            if not processed_team_data:
                logger.error("Failed to process team data")
                return None
                
            # Store team data
            team = self.storage.store_team_data(processed_team_data)
            return processed_team_data
            
        except Exception as e:
            logger.error(f"Error collecting team data: {str(e)}")
            return None
            
    def collect_season_data(self, season: str) -> List[Dict]:
        """Collect and store data for an entire season."""
        try:
            # Get all match URLs for the season
            match_urls = self.scraper.get_season_match_urls(season)
            if not match_urls:
                logger.error(f"Failed to get match URLs for season {season}")
                return []
                
            collected_matches = []
            for match_url in match_urls:
                match_data = self.collect_match_data(match_url)
                if match_data:
                    collected_matches.append(match_data)
                    
            return collected_matches
            
        except Exception as e:
            logger.error(f"Error collecting season data: {str(e)}")
            return []
            
    def update_recent_data(self, days: int = 30) -> List[Dict]:
        """Update data for recent matches."""
        try:
            # Get recent matches from storage
            recent_matches = self.storage.get_recent_matches(days)
            if not recent_matches:
                logger.info(f"No recent matches found in the last {days} days")
                return []
                
            updated_matches = []
            for match in recent_matches:
                match_url = self.scraper.get_match_url(match.match_number, match.season)
                if match_url:
                    match_data = self.collect_match_data(match_url)
                    if match_data:
                        updated_matches.append(match_data)
                        
            return updated_matches
            
        except Exception as e:
            logger.error(f"Error updating recent data: {str(e)}")
            return []
            
    def cleanup(self):
        """Clean up resources."""
        try:
            self.scraper.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}") 