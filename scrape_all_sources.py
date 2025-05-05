"""
Script to run the unified IPL scraper.
"""

import logging
from pathlib import Path
from ipl_data.scrapers.unified_scraper import UnifiedIPLScraper

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('scraper.log')
        ]
    )

def main():
    """Main function to run the scraper."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create data directory
    data_dir = Path("ipl_data/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache directory
    cache_dir = Path("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize unified scraper
        scraper = UnifiedIPLScraper(cache_dir, data_dir)
        
        # Scrape and merge data
        logger.info("Starting scraping and merging data")
        scraper.scrape_and_merge()
        logger.info("Successfully completed scraping and merging data")
        
    except Exception as e:
        logger.error(f"Failed to scrape data: {e}")
        raise
    finally:
        # Cleanup
        if 'scraper' in locals():
            scraper.close()

if __name__ == "__main__":
    main() 