"""
ESPN Cricinfo scraper implementation.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from selenium.webdriver.common.by import By
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

class ESPNCricinfoScraper(BaseScraper):
    """Scraper for ESPN Cricinfo website."""
    
    def __init__(self, proxy_list: Optional[List[str]] = None):
        """Initialize the ESPN Cricinfo scraper."""
        super().__init__(proxy_list)
        self.base_url = "https://www.espncricinfo.com"
        
    def scrape_match_data(self, match_url: str) -> Dict:
        """Scrape data for a specific match."""
        try:
            self.setup_driver()
            self.safe_get(match_url)
            
            # Wait for match details to load
            self.wait_for_element(By.CLASS_NAME, "match-info")
            
            # Extract match data
            match_data = {
                'match_number': self._get_match_number(),
                'season': self._get_season(),
                'date': self._get_match_date(),
                'venue': self._get_venue(),
                'teams': self._get_teams(),
                'toss_winner': self._get_toss_winner(),
                'winner': self._get_winner(),
                'scores': self._get_scores(),
                'player_of_match': self._get_player_of_match(),
                'weather_condition': self._get_weather_condition(),
                'pitch_condition': self._get_pitch_condition(),
                'player_performances': self._get_player_performances()
            }
            
            return match_data
            
        except Exception as e:
            logger.error(f"Error scraping match data: {str(e)}")
            raise
        finally:
            self.cleanup()
            
    def scrape_player_data(self, player_url: str) -> Dict:
        """Scrape data for a specific player."""
        try:
            self.setup_driver()
            self.safe_get(player_url)
            
            # Wait for player details to load
            self.wait_for_element(By.CLASS_NAME, "player-overview")
            
            # Extract player data
            player_data = {
                'name': self._get_player_name(),
                'team': self._get_player_team(),
                'role': self._get_player_role(),
                'nationality': self._get_player_nationality(),
                'date_of_birth': self._get_player_dob(),
                'batting_stats': self._get_batting_stats(),
                'bowling_stats': self._get_bowling_stats(),
                'recent_performances': self._get_recent_performances()
            }
            
            return player_data
            
        except Exception as e:
            logger.error(f"Error scraping player data: {str(e)}")
            raise
        finally:
            self.cleanup()
            
    def scrape_team_data(self, team_url: str) -> Dict:
        """Scrape data for a specific team."""
        try:
            self.setup_driver()
            self.safe_get(team_url)
            
            # Wait for team details to load
            self.wait_for_element(By.CLASS_NAME, "team-overview")
            
            # Extract team data
            team_data = {
                'name': self._get_team_name(),
                'short_name': self._get_team_short_name(),
                'founded_year': self._get_team_founded_year(),
                'home_ground': self._get_team_home_ground(),
                'total_matches': self._get_team_total_matches(),
                'wins': self._get_team_wins(),
                'losses': self._get_team_losses(),
                'win_percentage': self._get_team_win_percentage(),
                'current_squad': self._get_team_squad(),
                'recent_form': self._get_team_recent_form()
            }
            
            return team_data
            
        except Exception as e:
            logger.error(f"Error scraping team data: {str(e)}")
            raise
        finally:
            self.cleanup()
            
    def get_season_match_urls(self, season: str) -> List[str]:
        """Get URLs for all matches in a season."""
        try:
            self.setup_driver()
            season_url = f"{self.base_url}/series/indian-premier-league-{season}"
            self.safe_get(season_url)
            
            # Wait for match list to load
            self.wait_for_element(By.CLASS_NAME, "match-list")
            
            # Extract match URLs
            match_elements = self.driver.find_elements(By.CSS_SELECTOR, ".match-list-item a")
            return [element.get_attribute("href") for element in match_elements]
            
        except Exception as e:
            logger.error(f"Error getting season match URLs: {str(e)}")
            return []
        finally:
            self.cleanup()
            
    def get_match_url(self, match_number: str) -> Optional[str]:
        """Get URL for a specific match number."""
        try:
            self.setup_driver()
            search_url = f"{self.base_url}/search?query=IPL+{match_number}"
            self.safe_get(search_url)
            
            # Wait for search results
            self.wait_for_element(By.CLASS_NAME, "search-results")
            
            # Find match URL
            match_elements = self.driver.find_elements(By.CSS_SELECTOR, ".search-results-item a")
            for element in match_elements:
                if match_number in element.text:
                    return element.get_attribute("href")
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting match URL: {str(e)}")
            return None
        finally:
            self.cleanup()
            
    # Helper methods for match data
    def _get_match_number(self) -> str:
        """Extract match number from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "match-number")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_season(self) -> str:
        """Extract season from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "season")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_match_date(self) -> str:
        """Extract match date from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "match-date")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_venue(self) -> str:
        """Extract venue from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "venue")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_teams(self) -> List[str]:
        """Extract teams from the page."""
        try:
            elements = self.driver.find_elements(By.CLASS_NAME, "team-name")
            return [element.text.strip() for element in elements]
        except Exception:
            return []
            
    def _get_toss_winner(self) -> str:
        """Extract toss winner from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "toss-winner")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_winner(self) -> str:
        """Extract match winner from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "match-winner")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_scores(self) -> Dict:
        """Extract scores from the page."""
        try:
            scores = {}
            score_elements = self.driver.find_elements(By.CLASS_NAME, "team-score")
            for element in score_elements:
                team = element.find_element(By.CLASS_NAME, "team-name").text.strip()
                score = element.find_element(By.CLASS_NAME, "score").text.strip()
                scores[team] = score
            return scores
        except Exception:
            return {}
            
    def _get_player_of_match(self) -> str:
        """Extract player of the match from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "player-of-match")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_weather_condition(self) -> str:
        """Extract weather condition from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "weather-condition")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_pitch_condition(self) -> str:
        """Extract pitch condition from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "pitch-condition")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_player_performances(self) -> List[Dict]:
        """Extract player performances from the page."""
        try:
            performances = []
            performance_elements = self.driver.find_elements(By.CLASS_NAME, "player-performance")
            
            for element in performance_elements:
                performance = {
                    'player_name': element.find_element(By.CLASS_NAME, "player-name").text.strip(),
                    'runs_scored': element.find_element(By.CLASS_NAME, "runs").text.strip(),
                    'balls_faced': element.find_element(By.CLASS_NAME, "balls").text.strip(),
                    'fours': element.find_element(By.CLASS_NAME, "fours").text.strip(),
                    'sixes': element.find_element(By.CLASS_NAME, "sixes").text.strip(),
                    'wickets_taken': element.find_element(By.CLASS_NAME, "wickets").text.strip(),
                    'overs_bowled': element.find_element(By.CLASS_NAME, "overs").text.strip(),
                    'runs_conceded': element.find_element(By.CLASS_NAME, "runs-conceded").text.strip(),
                    'maidens': element.find_element(By.CLASS_NAME, "maidens").text.strip(),
                    'catches': element.find_element(By.CLASS_NAME, "catches").text.strip(),
                    'stumpings': element.find_element(By.CLASS_NAME, "stumpings").text.strip()
                }
                performances.append(performance)
                
            return performances
            
        except Exception:
            return []
            
    # Helper methods for player data
    def _get_player_name(self) -> str:
        """Extract player name from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "player-name")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_player_team(self) -> str:
        """Extract player's team from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "player-team")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_player_role(self) -> str:
        """Extract player's role from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "player-role")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_player_nationality(self) -> str:
        """Extract player's nationality from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "player-nationality")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_player_dob(self) -> str:
        """Extract player's date of birth from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "player-dob")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_batting_stats(self) -> Dict:
        """Extract player's batting statistics from the page."""
        try:
            stats = {}
            stat_elements = self.driver.find_elements(By.CLASS_NAME, "batting-stat")
            for element in stat_elements:
                name = element.find_element(By.CLASS_NAME, "stat-name").text.strip()
                value = element.find_element(By.CLASS_NAME, "stat-value").text.strip()
                stats[name] = value
            return stats
        except Exception:
            return {}
            
    def _get_bowling_stats(self) -> Dict:
        """Extract player's bowling statistics from the page."""
        try:
            stats = {}
            stat_elements = self.driver.find_elements(By.CLASS_NAME, "bowling-stat")
            for element in stat_elements:
                name = element.find_element(By.CLASS_NAME, "stat-name").text.strip()
                value = element.find_element(By.CLASS_NAME, "stat-value").text.strip()
                stats[name] = value
            return stats
        except Exception:
            return {}
            
    def _get_recent_performances(self) -> List[Dict]:
        """Extract player's recent performances from the page."""
        try:
            performances = []
            performance_elements = self.driver.find_elements(By.CLASS_NAME, "recent-performance")
            
            for element in performance_elements:
                performance = {
                    'match': element.find_element(By.CLASS_NAME, "match").text.strip(),
                    'runs': element.find_element(By.CLASS_NAME, "runs").text.strip(),
                    'wickets': element.find_element(By.CLASS_NAME, "wickets").text.strip(),
                    'catches': element.find_element(By.CLASS_NAME, "catches").text.strip()
                }
                performances.append(performance)
                
            return performances
            
        except Exception:
            return []
            
    # Helper methods for team data
    def _get_team_name(self) -> str:
        """Extract team name from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "team-name")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_team_short_name(self) -> str:
        """Extract team's short name from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "team-short-name")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_team_founded_year(self) -> int:
        """Extract team's founded year from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "team-founded-year")
            return int(element.text.strip())
        except Exception:
            return 2008  # Default to IPL's first year
            
    def _get_team_home_ground(self) -> str:
        """Extract team's home ground from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "team-home-ground")
            return element.text.strip()
        except Exception:
            return ""
            
    def _get_team_total_matches(self) -> int:
        """Extract team's total matches from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "team-total-matches")
            return int(element.text.strip())
        except Exception:
            return 0
            
    def _get_team_wins(self) -> int:
        """Extract team's wins from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "team-wins")
            return int(element.text.strip())
        except Exception:
            return 0
            
    def _get_team_losses(self) -> int:
        """Extract team's losses from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "team-losses")
            return int(element.text.strip())
        except Exception:
            return 0
            
    def _get_team_win_percentage(self) -> float:
        """Extract team's win percentage from the page."""
        try:
            element = self.driver.find_element(By.CLASS_NAME, "team-win-percentage")
            return float(element.text.strip().rstrip('%'))
        except Exception:
            return 0.0
            
    def _get_team_squad(self) -> List[str]:
        """Extract team's current squad from the page."""
        try:
            elements = self.driver.find_elements(By.CLASS_NAME, "team-squad-player")
            return [element.text.strip() for element in elements]
        except Exception:
            return []
            
    def _get_team_recent_form(self) -> List[str]:
        """Extract team's recent form from the page."""
        try:
            elements = self.driver.find_elements(By.CLASS_NAME, "team-recent-form")
            return [element.text.strip() for element in elements]
        except Exception:
            return [] 