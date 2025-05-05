"""
ESPNCricinfo scraper implementation for IPL data.
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

from .base_scraper import BaseScraper
from ..config.config import ESPNCRICINFO_BASE_URL, ESPNCRICINFO_SELECTORS

class ESPNCricinfoScraper(BaseScraper):
    """Scraper for ESPNCricinfo website."""
    
    def __init__(self, cache_dir: str = "cache", data_dir: str = "ipl_data/data/raw"):
        """Initialize ESPNCricinfo scraper."""
        super().__init__(cache_dir)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Base URLs
        self.base_url = "https://www.espncricinfo.com"
        self.stats_url = f"{self.base_url}/series/indian-premier-league-2024-1410320/stats"
        self.matches_url = f"{self.base_url}/series/indian-premier-league-2024-1410320/matches"
        self.points_url = f"{self.base_url}/series/indian-premier-league-2024-1410320/points-table"
        self.teams_url = f"{self.base_url}/series/indian-premier-league-2024-1410320/teams"
        
    def scrape_player_stats(self) -> pd.DataFrame:
        """
        Scrape player statistics.
        
        Returns:
            DataFrame with player stats
        """
        self.logger.info(f"Scraping player stats from {self.stats_url}")
        
        try:
            soup = self.get(self.stats_url)
            stats_container = soup.find('div', class_='ds-overflow-x-auto')
            
            if not stats_container:
                self.logger.warning("Stats container not found")
                return pd.DataFrame()
            
            # Extract batting stats
            batting_stats = []
            batting_table = stats_container.find('table', class_='ds-table')
            if batting_table:
                headers = [th.text.strip() for th in batting_table.find_all('th')]
                for row in batting_table.find_all('tr')[1:]:  # Skip header row
                    cells = row.find_all('td')
                    if cells:
                        stat = {
                            'player': cells[0].text.strip(),
                            'team': cells[1].text.strip(),
                            'matches': int(cells[2].text.strip()),
                            'innings': int(cells[3].text.strip()),
                            'runs': int(cells[4].text.strip()),
                            'average': float(cells[5].text.strip() or 0),
                            'strike_rate': float(cells[6].text.strip() or 0),
                            'fifties': int(cells[7].text.strip()),
                            'hundreds': int(cells[8].text.strip()),
                            'type': 'batting'
                        }
                        batting_stats.append(stat)
            
            # Extract bowling stats
            bowling_stats = []
            bowling_table = stats_container.find('table', class_='ds-table--bowling')
            if bowling_table:
                headers = [th.text.strip() for th in bowling_table.find_all('th')]
                for row in bowling_table.find_all('tr')[1:]:  # Skip header row
                    cells = row.find_all('td')
                    if cells:
                        stat = {
                            'player': cells[0].text.strip(),
                            'team': cells[1].text.strip(),
                            'matches': int(cells[2].text.strip()),
                            'innings': int(cells[3].text.strip()),
                            'overs': float(cells[4].text.strip()),
                            'wickets': int(cells[5].text.strip()),
                            'average': float(cells[6].text.strip() or 0),
                            'economy': float(cells[7].text.strip() or 0),
                            'type': 'bowling'
                        }
                        bowling_stats.append(stat)
            
            # Combine stats and create DataFrame
            all_stats = batting_stats + bowling_stats
            df = pd.DataFrame(all_stats)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"player_stats_espncricinfo_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved player stats to {output_file}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to scrape player stats: {e}")
            raise
    
    def scrape_match_schedule(self) -> pd.DataFrame:
        """
        Scrape match schedule.
        
        Returns:
            DataFrame with match schedule
        """
        self.logger.info(f"Scraping match schedule from {self.matches_url}")
        
        try:
            soup = self.get(self.matches_url)
            schedule_container = soup.find('div', class_='ds-flex')
            
            if not schedule_container:
                self.logger.warning("Match schedule container not found")
                return pd.DataFrame()
            
            matches = []
            for match in schedule_container.find_all('div', class_='ds-flex'):
                try:
                    teams = match.find_all('span', class_='ds-text-tight-m')
                    date_elem = match.find('span', class_='ds-text-tight-s')
                    time_elem = match.find('span', class_='ds-text-tight-xs')
                    venue_elem = match.find('span', class_='ds-text-tight-s')
                    status_elem = match.find('span', class_='ds-text-tight-m')
                    
                    match_data = {
                        'team1': teams[0].text.strip() if len(teams) > 0 else None,
                        'team2': teams[1].text.strip() if len(teams) > 1 else None,
                        'date': date_elem.text.strip() if date_elem else None,
                        'time': time_elem.text.strip() if time_elem else None,
                        'venue': venue_elem.text.strip() if venue_elem else None,
                        'status': status_elem.text.strip() if status_elem else None
                    }
                    matches.append(match_data)
                except (AttributeError, IndexError) as e:
                    self.logger.warning(f"Failed to parse match card: {e}")
                    continue
            
            df = pd.DataFrame(matches)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"match_schedule_espncricinfo_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved match schedule to {output_file}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to scrape match schedule: {e}")
            raise
    
    def scrape_points_table(self) -> pd.DataFrame:
        """
        Scrape points table.
        
        Returns:
            DataFrame with points table
        """
        self.logger.info(f"Scraping points table from {self.points_url}")
        
        try:
            soup = self.get(self.points_url)
            points_table = soup.find('table', class_='ds-table')
            
            if not points_table:
                self.logger.warning("Points table not found")
                return pd.DataFrame()
            
            # Extract headers
            headers = [th.text.strip() for th in points_table.find_all('th')]
            
            # Extract rows
            rows = []
            for tr in points_table.find_all('tr')[1:]:  # Skip header row
                cells = tr.find_all('td')
                if cells:
                    row = {
                        'team': cells[0].text.strip(),
                        'matches': int(cells[1].text.strip()),
                        'won': int(cells[2].text.strip()),
                        'lost': int(cells[3].text.strip()),
                        'tied': int(cells[4].text.strip()),
                        'no_result': int(cells[5].text.strip()),
                        'points': int(cells[6].text.strip()),
                        'net_run_rate': float(cells[7].text.strip())
                    }
                    rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"points_table_espncricinfo_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved points table to {output_file}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to scrape points table: {e}")
            raise
    
    def scrape_team_info(self) -> pd.DataFrame:
        """
        Scrape team information.
        
        Returns:
            DataFrame with team information
        """
        self.logger.info(f"Scraping team information from {self.teams_url}")
        
        try:
            soup = self.get(self.teams_url)
            teams_container = soup.find('div', class_='ds-flex')
            
            if not teams_container:
                self.logger.warning("Teams container not found")
                return pd.DataFrame()
            
            teams = []
            for team in teams_container.find_all('div', class_='ds-flex'):
                try:
                    name_elem = team.find('h2', class_='ds-text-tight-l')
                    captain_elem = team.find('div', class_='ds-text-tight-m')
                    coach_elem = team.find('div', class_='ds-text-tight-s')
                    home_elem = team.find('div', class_='ds-text-tight-s')
                    
                    team_data = {
                        'name': name_elem.text.strip() if name_elem else None,
                        'captain': captain_elem.text.strip() if captain_elem else None,
                        'coach': coach_elem.text.strip() if coach_elem else None,
                        'home_ground': home_elem.text.strip() if home_elem else None
                    }
                    teams.append(team_data)
                except AttributeError as e:
                    self.logger.warning(f"Failed to parse team card: {e}")
                    continue
            
            df = pd.DataFrame(teams)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"team_info_espncricinfo_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved team information to {output_file}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to scrape team information: {e}")
            raise
    
    def scrape_all(self):
        """Scrape all available data."""
        try:
            self.scrape_player_stats()
            self.scrape_match_schedule()
            self.scrape_points_table()
            self.scrape_team_info()
        except Exception as e:
            self.logger.error(f"Failed to scrape all data: {e}")
            raise 