"""
Test script for improved IPL scrapers.
"""

import logging
from pathlib import Path
import pandas as pd
from ipl_data.scrapers.cricbuzz import CricbuzzScraper

def setup_logging():
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "test_scraper.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_dataframe(df: pd.DataFrame, name: str, data_dir: Path):
    """Save DataFrame to CSV if not empty."""
    if not df.empty:
        data_dir.mkdir(parents=True, exist_ok=True)
        output_file = data_dir / f"{name}.csv"
        df.to_csv(output_file, index=False)
        return True
    return False

def test_cricbuzz_scraper():
    """Test the improved Cricbuzz scraper."""
    logger = setup_logging()
    data_dir = Path("data/raw")
    
    try:
        logger.info("Initializing Cricbuzz scraper...")
        scraper = CricbuzzScraper(cache_dir="cache")
        
        # Test player stats
        logger.info("Testing player statistics scraping...")
        stats_df = scraper.scrape_player_stats()
        if save_dataframe(stats_df, "player_stats_cricbuzz", data_dir):
            logger.info(f"Successfully scraped {len(stats_df)} player statistics")
            logger.info("\nPlayer Stats Preview:")
            logger.info(stats_df.head().to_string())
        else:
            logger.warning("No player statistics found")
        
        # Test match schedule
        logger.info("\nTesting match schedule scraping...")
        schedule_df = scraper.scrape_match_schedule()
        if save_dataframe(schedule_df, "match_schedule_cricbuzz", data_dir):
            logger.info(f"Successfully scraped {len(schedule_df)} matches")
            logger.info("\nMatch Schedule Preview:")
            logger.info(schedule_df.head().to_string())
        else:
            logger.warning("No match schedule found")
        
        # Test points table
        logger.info("\nTesting points table scraping...")
        points_df = scraper.scrape_points_table()
        if save_dataframe(points_df, "points_table_cricbuzz", data_dir):
            logger.info(f"Successfully scraped points table with {len(points_df)} teams")
            logger.info("\nPoints Table Preview:")
            logger.info(points_df.head().to_string())
        else:
            logger.warning("No points table found")
        
        # Test team info
        logger.info("\nTesting team information scraping...")
        teams_df = scraper.scrape_team_info()
        if save_dataframe(teams_df, "team_info_cricbuzz", data_dir):
            logger.info(f"Successfully scraped info for {len(teams_df)} teams")
            logger.info("\nTeam Info Preview:")
            logger.info(teams_df.head().to_string())
        else:
            logger.warning("No team information found")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
    finally:
        scraper.close()
        logger.info("Test completed")

if __name__ == "__main__":
    test_cricbuzz_scraper() 