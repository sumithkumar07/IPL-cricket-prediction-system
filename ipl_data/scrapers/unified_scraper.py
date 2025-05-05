"""
Unified scraper for IPL data from multiple sources.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .iplt20 import IPLt20Scraper
from .espncricinfo import ESPNCricinfoScraper
from .cricbuzz import CricbuzzScraper


class UnifiedIPLScraper:
    """Unified scraper for IPL data from multiple sources."""

    def __init__(self, cache_dir: str = "cache", data_dir: str = "ipl_data/data/raw"):
        """Initialize unified scraper."""
        self.cache_dir = Path(cache_dir)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize scrapers
        self.scrapers = {
            "iplt20": IPLt20Scraper(cache_dir, data_dir),
            "espncricinfo": ESPNCricinfoScraper(cache_dir, data_dir),
            "cricbuzz": CricbuzzScraper(cache_dir, data_dir),
        }

    def scrape_all_sources(self):
        """Scrape data from all sources concurrently."""
        self.logger.info("Starting scraping from all sources")

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit scraping tasks
            future_to_scraper = {
                executor.submit(self._scrape_source, name, scraper): name
                for name, scraper in self.scrapers.items()
            }

            # Process results as they complete
            for future in as_completed(future_to_scraper):
                scraper_name = future_to_scraper[future]
                try:
                    future.result()
                    self.logger.info(f"Successfully scraped data from {scraper_name}")
                except Exception as e:
                    self.logger.error(f"Failed to scrape data from {scraper_name}: {e}")

    def _scrape_source(self, name: str, scraper):
        """Scrape data from a single source."""
        self.logger.info(f"Scraping data from {name}")
        scraper.scrape_all()

    def merge_data(self):
        """Merge data from all sources."""
        self.logger.info("Merging data from all sources")

        # Merge player stats
        player_stats = []
        for name, scraper in self.scrapers.items():
            try:
                stats = scraper.scrape_player_stats()
                if not stats.empty:
                    stats["source"] = name
                    player_stats.append(stats)
            except Exception as e:
                self.logger.error(f"Failed to get player stats from {name}: {e}")

        if player_stats:
            merged_stats = pd.concat(player_stats, ignore_index=True)
            output_file = self.data_dir / "merged_player_stats.csv"
            merged_stats.to_csv(output_file, index=False)
            self.logger.info(f"Saved merged player stats to {output_file}")

        # Merge match schedules
        match_schedules = []
        for name, scraper in self.scrapers.items():
            try:
                schedule = scraper.scrape_match_schedule()
                if not schedule.empty:
                    schedule["source"] = name
                    match_schedules.append(schedule)
            except Exception as e:
                self.logger.error(f"Failed to get match schedule from {name}: {e}")

        if match_schedules:
            merged_schedule = pd.concat(match_schedules, ignore_index=True)
            output_file = self.data_dir / "merged_match_schedule.csv"
            merged_schedule.to_csv(output_file, index=False)
            self.logger.info(f"Saved merged match schedule to {output_file}")

        # Merge points tables
        points_tables = []
        for name, scraper in self.scrapers.items():
            try:
                points = scraper.scrape_points_table()
                if not points.empty:
                    points["source"] = name
                    points_tables.append(points)
            except Exception as e:
                self.logger.error(f"Failed to get points table from {name}: {e}")

        if points_tables:
            merged_points = pd.concat(points_tables, ignore_index=True)
            output_file = self.data_dir / "merged_points_table.csv"
            merged_points.to_csv(output_file, index=False)
            self.logger.info(f"Saved merged points table to {output_file}")

        # Merge team info
        team_infos = []
        for name, scraper in self.scrapers.items():
            try:
                teams = scraper.scrape_team_info()
                if not teams.empty:
                    teams["source"] = name
                    team_infos.append(teams)
            except Exception as e:
                self.logger.error(f"Failed to get team info from {name}: {e}")

        if team_infos:
            merged_teams = pd.concat(team_infos, ignore_index=True)
            output_file = self.data_dir / "merged_team_info.csv"
            merged_teams.to_csv(output_file, index=False)
            self.logger.info(f"Saved merged team info to {output_file}")

    def scrape_and_merge(self):
        """Scrape data from all sources and merge them."""
        self.scrape_all_sources()
        self.merge_data()

    def close(self):
        """Close all scrapers and cleanup resources."""
        for scraper in self.scrapers.values():
            scraper.close()
