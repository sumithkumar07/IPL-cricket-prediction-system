"""
Storage manager for handling database operations.
"""

import logging
from typing import Dict, List, Optional
from django.db import transaction
from django.db.models import Q
from predictions.models import (
    Team, Player, Match, PlayerPerformance
)
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StorageManager:
    """Class for managing database operations."""
    
    def __init__(self):
        """Initialize the storage manager."""
        self.batch_size = 100  # Number of records to process in a batch
        
    @transaction.atomic
    def store_match_data(self, match_data: Dict) -> Match:
        """Store match data in the database."""
        try:
            # Get or create teams
            team1 = self._get_or_create_team(match_data['teams'][0])
            team2 = self._get_or_create_team(match_data['teams'][1])
            
            # Get toss winner and match winner
            toss_winner = self._get_or_create_team(match_data['toss_winner'])
            winner = self._get_or_create_team(match_data['winner'])
            
            # Create match record
            match = Match.objects.create(
                match_number=match_data['match_number'],
                season=match_data['season'],
                date=match_data['date'],
                venue=match_data['venue'],
                team1=team1,
                team2=team2,
                toss_winner=toss_winner,
                winner=winner,
                team1_score=match_data['scores'][team1.name]['runs'],
                team1_wickets=match_data['scores'][team1.name]['wickets'],
                team2_score=match_data['scores'][team2.name]['runs'],
                team2_wickets=match_data['scores'][team2.name]['wickets'],
                player_of_match=match_data['player_of_match'],
                weather_condition=match_data['weather_condition'],
                pitch_condition=match_data['pitch_condition']
            )
            
            return match
            
        except Exception as e:
            logger.error(f"Error storing match data: {str(e)}")
            raise
            
    @transaction.atomic
    def store_player_data(self, player_data: Dict) -> Player:
        """Store player data in the database."""
        try:
            # Get team
            team = self._get_or_create_team(player_data['team'])
            
            # Create or update player record
            player, created = Player.objects.update_or_create(
                name=player_data['name'],
                defaults={
                    'team': team,
                    'role': player_data['role'],
                    'nationality': player_data['nationality'],
                    'date_of_birth': player_data['date_of_birth'],
                    'batting_stats': player_data['batting_stats'],
                    'bowling_stats': player_data['bowling_stats'],
                    'is_active': True
                }
            )
            
            return player
            
        except Exception as e:
            logger.error(f"Error storing player data: {str(e)}")
            raise
            
    @transaction.atomic
    def store_team_data(self, team_data: Dict) -> Team:
        """Store team data in the database."""
        try:
            # Create or update team record
            team, created = Team.objects.update_or_create(
                name=team_data['name'],
                defaults={
                    'short_name': team_data['short_name'],
                    'founded_year': team_data['founded_year'],
                    'home_ground': team_data['home_ground'],
                    'total_matches': team_data['total_matches'],
                    'wins': team_data['wins'],
                    'losses': team_data['losses'],
                    'win_percentage': team_data['win_percentage']
                }
            )
            
            return team
            
        except Exception as e:
            logger.error(f"Error storing team data: {str(e)}")
            raise
            
    def store_player_performance(self, performance_data: Dict) -> PlayerPerformance:
        """Store player performance data in the database."""
        try:
            # Get match and player
            match = Match.objects.get(match_number=performance_data['match_number'])
            player = Player.objects.get(name=performance_data['player_name'])
            
            # Create performance record
            performance = PlayerPerformance.objects.create(
                match=match,
                player=player,
                runs_scored=performance_data['runs_scored'],
                balls_faced=performance_data['balls_faced'],
                fours=performance_data['fours'],
                sixes=performance_data['sixes'],
                wickets_taken=performance_data['wickets_taken'],
                overs_bowled=performance_data['overs_bowled'],
                runs_conceded=performance_data['runs_conceded'],
                maidens=performance_data['maidens'],
                catches=performance_data['catches'],
                stumpings=performance_data['stumpings']
            )
            
            return performance
            
        except Exception as e:
            logger.error(f"Error storing player performance: {str(e)}")
            raise
            
    def _get_or_create_team(self, team_name: str) -> Team:
        """Get or create a team record."""
        try:
            team, created = Team.objects.get_or_create(
                name=team_name,
                defaults={
                    'short_name': team_name[:3].upper(),
                    'founded_year': 2008,  # Default to IPL's first year
                    'home_ground': '',
                    'total_matches': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_percentage': 0.0
                }
            )
            return team
        except Exception as e:
            logger.error(f"Error getting or creating team: {str(e)}")
            raise
            
    def get_recent_matches(self, days: int = 30) -> List[Match]:
        """Get recent matches within the specified number of days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            return Match.objects.filter(date__gte=cutoff_date).order_by('-date')
        except Exception as e:
            logger.error(f"Error getting recent matches: {str(e)}")
            return []
            
    def get_team_matches(self, team_name: str, season: Optional[str] = None) -> List[Match]:
        """Get matches for a specific team and optional season."""
        try:
            team = Team.objects.get(name=team_name)
            query = Q(team1=team) | Q(team2=team)
            
            if season:
                query &= Q(season=season)
                
            return Match.objects.filter(query).order_by('-date')
        except Exception as e:
            logger.error(f"Error getting team matches: {str(e)}")
            return []
            
    def get_player_performances(
        self,
        player_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[PlayerPerformance]:
        """Get performances for a specific player within an optional date range."""
        try:
            player = Player.objects.get(name=player_name)
            query = Q(player=player)
            
            if start_date:
                query &= Q(match__date__gte=start_date)
            if end_date:
                query &= Q(match__date__lte=end_date)
                
            return PlayerPerformance.objects.filter(query).order_by('-match__date')
        except Exception as e:
            logger.error(f"Error getting player performances: {str(e)}")
            return [] 