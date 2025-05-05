"""
Script to scrape data from all IPL data sources.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

from ipl_data.scrapers.iplt20 import IPLt20Scraper

def setup_logging():
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("Starting IPL data scraping...")
    
    try:
        # Initialize IPL scraper
        with IPLt20Scraper() as scraper:
            # Scrape player statistics for current season
            logger.info("Scraping player statistics...")
            player_stats = scraper.scrape_player_stats(season=2024)
            if not player_stats.empty:
                logger.info(f"Found {len(player_stats)} player records")
                print("\nPlayer Statistics Preview:")
                print(player_stats.head())
            
            time.sleep(2)  # Respect rate limits
            
            # Scrape match schedule
            logger.info("\nScraping match schedule...")
            schedule = scraper.scrape_match_schedule()
            if not schedule.empty:
                logger.info(f"Found {len(schedule)} scheduled matches")
                print("\nMatch Schedule Preview:")
                print(schedule.head())
            
            time.sleep(2)
            
            # Scrape venue details
            logger.info("\nScraping venue details...")
            venues = scraper.scrape_venue_details()
            if not venues.empty:
                logger.info(f"Found {len(venues)} venues")
                print("\nVenue Details Preview:")
                print(venues.head())
            
            time.sleep(2)
            
            # Scrape points table
            logger.info("\nScraping points table...")
            points = scraper.scrape_points_table()
            if not points.empty:
                logger.info(f"Found {len(points)} team records")
                print("\nPoints Table Preview:")
                print(points.head())
            
            time.sleep(2)
            
            # Scrape live scores
            logger.info("\nScraping live scores...")
            live_scores = scraper.scrape_live_scores()
            if not live_scores.empty:
                logger.info(f"Found {len(live_scores)} live matches")
                print("\nLive Scores Preview:")
                print(live_scores.head())
            
            # Print summary
            data_dir = Path("ipl-data/data/raw")
            if data_dir.exists():
                csv_files = list(data_dir.glob("*.csv"))
                logger.info(f"\nScraped data summary:")
                logger.info(f"Total files: {len(csv_files)}")
                for file in csv_files:
                    size = file.stat().st_size / 1024  # Size in KB
                    logger.info(f"- {file.name}: {size:.2f} KB")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise
    
    logger.info("Scraping completed!")

if __name__ == "__main__":
    main() 