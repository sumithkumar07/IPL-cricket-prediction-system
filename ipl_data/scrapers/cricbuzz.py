"""
Cricbuzz scraper implementation with improved selectors and error handling.
"""

import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from urllib.parse import urljoin

from .base import BaseScraper

class CricbuzzScraper(BaseScraper):
    """Scraper for Cricbuzz with improved selectors."""
    
    # Updated selectors for 2024 IPL season
    SELECTORS = {
        "stats": {
            "container": "div.cb-col.cb-col-100.cb-ltst-wgt-hdr",
            "batting_table": "table.cb-pnl-tbl.cb-stats-mtr",
            "bowling_table": "table.cb-pnl-tbl.cb-stats-mtr",
            "player_row": "tr.cb-srs-stats-tr",
            "player_name": "a.cb-text-link",
            "team_name": "span.text-gray",
            "stats_cell": "td.cb-srs-stats-td"
        },
        "matches": {
            "container": "div.cb-col.cb-col-100.cb-series-matches",
            "match_card": "div.cb-mtch-lst.cb-col.cb-col-100.cb-tms-itm",
            "match_info": "div.cb-col-60.cb-col.cb-mtch-info",
            "teams": "h3.cb-text-gray",
            "venue_date": "div.cb-col-40.cb-col.text-gray",
            "match_number": "div.cb-col-40.cb-col.cb-mtch-numb"
        },
        "points": {
            "container": "div.cb-col.cb-col-100.cb-series-points",
            "table": "table.cb-srs-pnts",
            "team_row": "tr.cb-srs-pnts-row",
            "team_name": "td.cb-srs-pnts-name",
            "stats_cell": "td.cb-srs-pnts-td"
        },
        "teams": {
            "container": "div.cb-col.cb-col-100.cb-series-teams",
            "team_card": "div.cb-col.cb-col-50.cb-team-card",
            "team_name": "h2.cb-team-name",
            "captain": "div.cb-team-captain",
            "coach": "div.cb-team-coach",
            "home": "div.cb-team-ground"
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize Cricbuzz scraper."""
        super().__init__(cache_dir)
        
        # Set up specific headers for Cricbuzz
        self.session.headers.update({
            "Host": "www.cricbuzz.com",
            "Referer": "https://www.cricbuzz.com/cricket-series/indian-premier-league-2024",
            "Sec-Fetch-Site": "same-origin"
        })
        
        # Base URLs
        self.base_url = "https://www.cricbuzz.com"
        self.ipl_url = f"{self.base_url}/cricket-series/2375/indian-premier-league-2024"
    
    def _validate_data(self, df: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
        """Validate and clean DataFrame."""
        if df.empty:
            return df
        
        # Check for required columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing columns: {missing_cols}")
            return pd.DataFrame()
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        return df
    
    def scrape_player_stats(self) -> pd.DataFrame:
        """Scrape player statistics with improved error handling."""
        try:
            stats_url = f"{self.ipl_url}/stats"
            soup = self._make_request(stats_url)
            
            # Find stats container
            stats_container = soup.select_one(self.SELECTORS["stats"]["container"])
            if not stats_container:
                self.logger.error("Stats container not found")
                return pd.DataFrame()
            
            # Process batting stats
            batting_stats = []
            batting_table = stats_container.select_one(self.SELECTORS["stats"]["batting_table"])
            if batting_table:
                for row in batting_table.select(self.SELECTORS["stats"]["player_row"]):
                    try:
                        cells = row.select(self.SELECTORS["stats"]["stats_cell"])
                        if len(cells) >= 8:
                            player_name = row.select_one(self.SELECTORS["stats"]["player_name"])
                            team_name = row.select_one(self.SELECTORS["stats"]["team_name"])
                            
                            stat = {
                                "name": player_name.text.strip() if player_name else "Unknown",
                                "team": team_name.text.strip() if team_name else "Unknown",
                                "matches": cells[0].text.strip(),
                                "innings": cells[1].text.strip(),
                                "runs": cells[2].text.strip(),
                                "average": cells[3].text.strip(),
                                "strike_rate": cells[4].text.strip(),
                                "type": "batting"
                            }
                            batting_stats.append(stat)
                    except Exception as e:
                        self.logger.warning(f"Error parsing batting row: {e}")
                        continue
            
            # Process bowling stats
            bowling_stats = []
            bowling_table = stats_container.select_one(self.SELECTORS["stats"]["bowling_table"])
            if bowling_table:
                for row in bowling_table.select(self.SELECTORS["stats"]["player_row"]):
                    try:
                        cells = row.select(self.SELECTORS["stats"]["stats_cell"])
                        if len(cells) >= 8:
                            player_name = row.select_one(self.SELECTORS["stats"]["player_name"])
                            team_name = row.select_one(self.SELECTORS["stats"]["team_name"])
                            
                            stat = {
                                "name": player_name.text.strip() if player_name else "Unknown",
                                "team": team_name.text.strip() if team_name else "Unknown",
                                "matches": cells[0].text.strip(),
                                "overs": cells[1].text.strip(),
                                "wickets": cells[2].text.strip(),
                                "economy": cells[3].text.strip(),
                                "average": cells[4].text.strip(),
                                "type": "bowling"
                            }
                            bowling_stats.append(stat)
                    except Exception as e:
                        self.logger.warning(f"Error parsing bowling row: {e}")
                        continue
            
            # Combine and validate data
            all_stats = batting_stats + bowling_stats
            df = pd.DataFrame(all_stats)
            
            expected_columns = ["name", "team", "matches", "type"]
            return self._validate_data(df, expected_columns)
            
        except Exception as e:
            self.logger.error(f"Error scraping player stats: {e}")
            return pd.DataFrame()
    
    def scrape_match_schedule(self) -> pd.DataFrame:
        """Scrape match schedule with improved error handling."""
        try:
            schedule_url = f"{self.ipl_url}/matches"
            soup = self._make_request(schedule_url)
            
            # Find schedule container
            container = soup.select_one(self.SELECTORS["matches"]["container"])
            if not container:
                self.logger.error("Schedule container not found")
                return pd.DataFrame()
            
            matches = []
            for card in container.select(self.SELECTORS["matches"]["match_card"]):
                try:
                    match_info = card.select_one(self.SELECTORS["matches"]["match_info"])
                    teams = match_info.select(self.SELECTORS["matches"]["teams"])
                    venue_date = card.select_one(self.SELECTORS["matches"]["venue_date"])
                    match_number = card.select_one(self.SELECTORS["matches"]["match_number"])
                    
                    if teams and venue_date:
                        venue_date_parts = venue_date.text.strip().split(",", 1)
                        matches.append({
                            "match_number": match_number.text.strip() if match_number else "Unknown",
                            "team1": teams[0].text.strip() if len(teams) > 0 else "TBD",
                            "team2": teams[1].text.strip() if len(teams) > 1 else "TBD",
                            "venue": venue_date_parts[0].strip() if len(venue_date_parts) > 0 else "Unknown",
                            "date": venue_date_parts[1].strip() if len(venue_date_parts) > 1 else "Unknown"
                        })
                except Exception as e:
                    self.logger.warning(f"Error parsing match card: {e}")
                    continue
            
            df = pd.DataFrame(matches)
            expected_columns = ["match_number", "team1", "team2", "venue", "date"]
            return self._validate_data(df, expected_columns)
            
        except Exception as e:
            self.logger.error(f"Error scraping match schedule: {e}")
        return pd.DataFrame()
    
    def scrape_points_table(self) -> pd.DataFrame:
        """Scrape points table with improved error handling."""
        try:
            points_url = f"{self.ipl_url}/points-table"
            soup = self._make_request(points_url)
            
            # Find points table container
            container = soup.select_one(self.SELECTORS["points"]["container"])
            if not container:
                self.logger.error("Points table container not found")
                return pd.DataFrame()
            
            teams = []
            table = container.select_one(self.SELECTORS["points"]["table"])
            if table:
                for row in table.select(self.SELECTORS["points"]["team_row"]):
                    try:
                        cells = row.select(self.SELECTORS["points"]["stats_cell"])
                        team_name = row.select_one(self.SELECTORS["points"]["team_name"])
                        
                        if team_name and len(cells) >= 6:
                            teams.append({
                                "team": team_name.text.strip(),
                                "matches": cells[0].text.strip(),
                                "won": cells[1].text.strip(),
                                "lost": cells[2].text.strip(),
                                "points": cells[3].text.strip(),
                                "nrr": cells[4].text.strip()
                            })
                    except Exception as e:
                        self.logger.warning(f"Error parsing team row: {e}")
                        continue
            
            df = pd.DataFrame(teams)
            expected_columns = ["team", "matches", "won", "lost", "points", "nrr"]
            return self._validate_data(df, expected_columns)
            
        except Exception as e:
            self.logger.error(f"Error scraping points table: {e}")
        return pd.DataFrame()
    
    def scrape_team_info(self) -> pd.DataFrame:
        """Scrape team information with improved error handling."""
        try:
            teams_url = f"{self.ipl_url}/teams"
            soup = self._make_request(teams_url)
            
            # Find teams container
            container = soup.select_one(self.SELECTORS["teams"]["container"])
            if not container:
                self.logger.error("Teams container not found")
                return pd.DataFrame()
            
            teams = []
            for card in container.select(self.SELECTORS["teams"]["team_card"]):
                try:
                    name = card.select_one(self.SELECTORS["teams"]["team_name"])
                    captain = card.select_one(self.SELECTORS["teams"]["captain"])
                    coach = card.select_one(self.SELECTORS["teams"]["coach"])
                    home = card.select_one(self.SELECTORS["teams"]["home"])
                    
                    teams.append({
                        "name": name.text.strip() if name else "Unknown",
                        "captain": captain.text.replace("Captain: ", "").strip() if captain else "Unknown",
                        "coach": coach.text.replace("Coach: ", "").strip() if coach else "Unknown",
                        "home_ground": home.text.strip() if home else "Unknown"
                    })
                except Exception as e:
                    self.logger.warning(f"Error parsing team card: {e}")
                    continue
            
            df = pd.DataFrame(teams)
            expected_columns = ["name", "captain", "coach", "home_ground"]
            return self._validate_data(df, expected_columns)
            
        except Exception as e:
            self.logger.error(f"Error scraping team info: {e}")
        return pd.DataFrame()
    
    def scrape_all(self) -> bool:
        """Scrape all data with proper error handling."""
        success = True
        try:
            stats_df = self.scrape_player_stats()
            schedule_df = self.scrape_match_schedule()
            points_df = self.scrape_points_table()
            teams_df = self.scrape_team_info()
            
            # Check if any scraping failed
            if any(df.empty for df in [stats_df, schedule_df, points_df, teams_df]):
                self.logger.warning("Some data could not be scraped")
                success = False
            
            return success
        except Exception as e:
            self.logger.error(f"Error in scrape_all: {e}")
            return False 