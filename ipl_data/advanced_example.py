"""
Advanced example usage of the IPL data scrapers with all features enabled.
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scrapers.iplt20 import IPLt20Scraper
from config.settings import PROXY_CONFIG, SELENIUM_CONFIG


def setup_proxies():
    """Setup proxy list from environment or file."""
    # You can add your proxies here or load them from a file
    PROXY_CONFIG["proxy_list"] = [
        # Add your proxy list here
        # Example: 'http://user:pass@192.168.1.1:8080'
    ]


def setup_chrome_driver():
    """Setup Chrome WebDriver path."""
    # Try to find Chrome WebDriver in common locations
    possible_paths = [
        "./chromedriver.exe",
        "C:/Program Files/Google/Chrome/Application/chromedriver.exe",
        "C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            SELENIUM_CONFIG["driver_path"] = path
            print(f"Found Chrome WebDriver at: {path}")
            return

    print(
        "Chrome WebDriver not found in common locations. Will attempt automatic download."
    )


def main():
    # Configure basic logging
    logging.basicConfig(level=logging.INFO)

    # Setup proxies if available
    setup_proxies()

    # Setup Chrome WebDriver
    setup_chrome_driver()

    print("\nInitializing IPL scraper...")
    ipl_scraper = IPLt20Scraper()

    try:
        # Scrape current season's player stats
        print("\nScraping player statistics for 2024...")
        player_stats = ipl_scraper.scrape_player_stats(season=2024)
        if player_stats is not None:
            print(f"Successfully scraped {len(player_stats)} player statistics")
            print("\nSample data:")
            print(player_stats.head())
        else:
            print("Failed to scrape player statistics")

        # Add delay between requests
        time.sleep(2)

        # Scrape match schedule
        print("\nScraping match schedule...")
        schedule = ipl_scraper.scrape_match_schedule()
        if schedule is not None:
            print(f"Successfully scraped {len(schedule)} matches")
            print("\nSample data:")
            print(schedule.head())
        else:
            print("Failed to scrape match schedule")

        # Add delay between requests
        time.sleep(2)

        # Scrape venue details
        print("\nScraping venue details...")
        venues = ipl_scraper.scrape_venue_details()
        if venues is not None:
            print(f"Successfully scraped {len(venues)} venues")
            print("\nSample data:")
            for venue in venues[:3]:  # Show first 3 venues
                print(venue)
        else:
            print("Failed to scrape venue details")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        logging.error("Error details:", exc_info=True)

    finally:
        print("\nCleaning up resources...")
        # Cleanup resources
        ipl_scraper.cleanup()

        # Print cache statistics
        cache_dir = Path("cache")
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.pkl"))
            print(f"\nCache statistics:")
            print(f"Total cached items: {len(cache_files)}")
            print(
                f"Cache size: {sum(f.stat().st_size for f in cache_files) / 1024:.2f} KB"
            )

        # Print log file location
        log_dir = Path("logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            print(f"\nLog files:")
            for log_file in log_files:
                print(f"- {log_file}")

        print("\nDone!")


if __name__ == "__main__":
    main()
