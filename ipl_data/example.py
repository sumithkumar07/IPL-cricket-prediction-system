"""
Example usage of the IPL data scrapers.
"""

from scrapers.iplt20 import IPLt20Scraper


def main():
    # Initialize the IPL scraper
    ipl_scraper = IPLt20Scraper()

    # Scrape current season's player stats
    print("Scraping player statistics for 2024...")
    player_stats = ipl_scraper.scrape_player_stats(season=2024)
    if player_stats is not None:
        print(f"Successfully scraped {len(player_stats)} player statistics")

    # Scrape match schedule
    print("\nScraping match schedule...")
    schedule = ipl_scraper.scrape_match_schedule()
    if schedule is not None:
        print(f"Successfully scraped {len(schedule)} matches")

    # Scrape venue details
    print("\nScraping venue details...")
    venues = ipl_scraper.scrape_venue_details()
    if venues is not None:
        print(f"Successfully scraped {len(venues)} venues")


if __name__ == "__main__":
    main()
