"""
Test script for Cricbuzz scraper.
"""

import logging
from pathlib import Path
from ipl_data.scrapers.cricbuzz import CricbuzzScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run the Cricbuzz scraper test."""
    logger = logging.getLogger(__name__)
    logger.info("Starting Cricbuzz scraper test")
    
    # Create data directory
    data_dir = Path("ipl_data/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize scraper
    scraper = CricbuzzScraper()
    
    try:
        # Test player stats
        logger.info("Testing player stats scraping...")
        stats_df = scraper.scrape_player_stats()
        if not stats_df.empty:
            logger.info(f"Successfully scraped {len(stats_df)} player stats")
        else:
            logger.warning("No player stats found")
        
        # Test match schedule
        logger.info("Testing match schedule scraping...")
        schedule_df = scraper.scrape_match_schedule()
        if not schedule_df.empty:
            logger.info(f"Successfully scraped {len(schedule_df)} matches")
        else:
            logger.warning("No match schedule found")
        
        # Test points table
        logger.info("Testing points table scraping...")
        points_df = scraper.scrape_points_table()
        if not points_df.empty:
            logger.info(f"Successfully scraped points table with {len(points_df)} teams")
        else:
            logger.warning("No points table found")
        
        # Test team info
        logger.info("Testing team info scraping...")
        teams_df = scraper.scrape_team_info()
        if not teams_df.empty:
            logger.info(f"Successfully scraped info for {len(teams_df)} teams")
        else:
            logger.warning("No team info found")
        
        logger.info("Cricbuzz scraper test completed")
        
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 