"""
Data processor for cleaning and normalizing scraped IPL data.
"""

import logging
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for processing and normalizing scraped IPL data."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.team_name_mapping = {
            'MI': 'Mumbai Indians',
            'CSK': 'Chennai Super Kings',
            'RCB': 'Royal Challengers Bangalore',
            'KKR': 'Kolkata Knight Riders',
            'DC': 'Delhi Capitals',
            'SRH': 'Sunrisers Hyderabad',
            'PBKS': 'Punjab Kings',
            'RR': 'Rajasthan Royals'
        }
        
    def process_match_data(self, match_data: Dict) -> Dict:
        """Process and normalize match data."""
        try:
            processed_data = match_data.copy()
            
            # Normalize team names
            processed_data['teams'] = [
                self._normalize_team_name(team)
                for team in processed_data.get('teams', [])
            ]
            
            # Normalize toss winner and match winner
            if 'toss_winner' in processed_data:
                processed_data['toss_winner'] = self._normalize_team_name(
                    processed_data['toss_winner']
                )
            if 'winner' in processed_data:
                processed_data['winner'] = self._normalize_team_name(
                    processed_data['winner']
                )
                
            # Convert date string to datetime
            if 'date' in processed_data:
                processed_data['date'] = self._parse_date(processed_data['date'])
                
            # Normalize scores
            if 'scores' in processed_data:
                processed_data['scores'] = self._normalize_scores(
                    processed_data['scores']
                )
                
            # Clean and validate data
            self._validate_match_data(processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing match data: {str(e)}")
            raise
            
    def process_player_data(self, player_data: Dict) -> Dict:
        """Process and normalize player data."""
        try:
            processed_data = player_data.copy()
            
            # Normalize team name
            if 'team' in processed_data:
                processed_data['team'] = self._normalize_team_name(
                    processed_data['team']
                )
                
            # Convert date of birth to datetime
            if 'date_of_birth' in processed_data:
                processed_data['date_of_birth'] = self._parse_date(
                    processed_data['date_of_birth']
                )
                
            # Normalize statistics
            if 'batting_stats' in processed_data:
                processed_data['batting_stats'] = self._normalize_stats(
                    processed_data['batting_stats']
                )
            if 'bowling_stats' in processed_data:
                processed_data['bowling_stats'] = self._normalize_stats(
                    processed_data['bowling_stats']
                )
                
            # Clean and validate data
            self._validate_player_data(processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing player data: {str(e)}")
            raise
            
    def process_team_data(self, team_data: Dict) -> Dict:
        """Process and normalize team data."""
        try:
            processed_data = team_data.copy()
            
            # Normalize team name
            if 'name' in processed_data:
                processed_data['name'] = self._normalize_team_name(
                    processed_data['name']
                )
                
            # Convert numeric fields
            numeric_fields = [
                'founded_year', 'total_matches', 'wins', 'losses',
                'win_percentage'
            ]
            for field in numeric_fields:
                if field in processed_data:
                    processed_data[field] = self._parse_numeric(
                        processed_data[field]
                    )
                    
            # Clean and validate data
            self._validate_team_data(processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing team data: {str(e)}")
            raise
            
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team name to standard format."""
        if not team_name:
            return ""
            
        # Remove extra whitespace
        team_name = team_name.strip()
        
        # Check if it's a short name
        if team_name in self.team_name_mapping:
            return self.team_name_mapping[team_name]
            
        # Check if it's a full name
        for short_name, full_name in self.team_name_mapping.items():
            if team_name.lower() == full_name.lower():
                return full_name
                
        return team_name
        
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        try:
            # Try different date formats
            formats = [
                '%Y-%m-%d',
                '%d-%m-%Y',
                '%d/%m/%Y',
                '%B %d, %Y',
                '%b %d, %Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
                    
            raise ValueError(f"Could not parse date: {date_str}")
            
        except Exception as e:
            logger.warning(f"Error parsing date {date_str}: {str(e)}")
            return None
            
    def _normalize_scores(self, scores: Dict) -> Dict:
        """Normalize match scores."""
        try:
            normalized_scores = {}
            
            for team, score in scores.items():
                # Extract runs and wickets
                match = re.match(r'(\d+)/(\d+)', score)
                if match:
                    runs = int(match.group(1))
                    wickets = int(match.group(2))
                    normalized_scores[team] = {
                        'runs': runs,
                        'wickets': wickets
                    }
                else:
                    normalized_scores[team] = {
                        'runs': 0,
                        'wickets': 0
                    }
                    
            return normalized_scores
            
        except Exception as e:
            logger.warning(f"Error normalizing scores: {str(e)}")
            return {}
            
    def _normalize_stats(self, stats: Dict) -> Dict:
        """Normalize player statistics."""
        try:
            normalized_stats = {}
            
            for key, value in stats.items():
                # Convert numeric values
                if isinstance(value, str):
                    try:
                        # Remove any non-numeric characters
                        clean_value = re.sub(r'[^\d.]', '', value)
                        if clean_value:
                            normalized_stats[key] = float(clean_value)
                        else:
                            normalized_stats[key] = 0
                    except ValueError:
                        normalized_stats[key] = value
                else:
                    normalized_stats[key] = value
                    
            return normalized_stats
            
        except Exception as e:
            logger.warning(f"Error normalizing stats: {str(e)}")
            return {}
            
    def _parse_numeric(self, value: Any) -> float:
        """Parse numeric value from string."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                # Remove any non-numeric characters
                clean_value = re.sub(r'[^\d.]', '', value)
                return float(clean_value) if clean_value else 0.0
            return 0.0
        except Exception:
            return 0.0
            
    def _validate_match_data(self, data: Dict) -> None:
        """Validate match data."""
        required_fields = [
            'match_number', 'season', 'date', 'venue',
            'teams', 'toss_winner', 'winner', 'scores'
        ]
        
        for field in required_fields:
            if field not in data or not data[field]:
                logger.warning(f"Missing required field in match data: {field}")
                
    def _validate_player_data(self, data: Dict) -> None:
        """Validate player data."""
        required_fields = [
            'name', 'team', 'role', 'nationality',
            'date_of_birth', 'batting_stats', 'bowling_stats'
        ]
        
        for field in required_fields:
            if field not in data or not data[field]:
                logger.warning(f"Missing required field in player data: {field}")
                
    def _validate_team_data(self, data: Dict) -> None:
        """Validate team data."""
        required_fields = [
            'name', 'short_name', 'founded_year',
            'home_ground', 'total_matches', 'wins',
            'losses', 'win_percentage'
        ]
        
        for field in required_fields:
            if field not in data or not data[field]:
                logger.warning(f"Missing required field in team data: {field}") 