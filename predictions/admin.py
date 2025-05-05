"""
Admin interface for IPL prediction system models.
"""

from django.contrib import admin
from .models import (
    Team, Player, Match, PlayerPerformance,
    Prediction, PlayerPrediction
)

@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ('name', 'short_name', 'founded_year', 'home_ground', 'win_percentage')
    search_fields = ('name', 'short_name', 'home_ground')
    list_filter = ('founded_year',)

@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ('name', 'team', 'role', 'nationality', 'is_active')
    search_fields = ('name', 'team__name', 'nationality')
    list_filter = ('role', 'nationality', 'is_active', 'team')

@admin.register(Match)
class MatchAdmin(admin.ModelAdmin):
    list_display = ('match_number', 'season', 'date', 'team1', 'team2', 'winner')
    search_fields = ('team1__name', 'team2__name', 'venue')
    list_filter = ('season', 'date', 'venue')

@admin.register(PlayerPerformance)
class PlayerPerformanceAdmin(admin.ModelAdmin):
    list_display = ('player', 'match', 'team', 'runs_scored', 'wickets_taken')
    search_fields = ('player__name', 'team__name', 'match__match_number')
    list_filter = ('match__season', 'match__date')

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('match', 'predicted_winner', 'confidence', 'model_version')
    search_fields = ('match__match_number', 'predicted_winner__name')
    list_filter = ('model_version', 'match__season')

@admin.register(PlayerPrediction)
class PlayerPredictionAdmin(admin.ModelAdmin):
    list_display = ('player', 'prediction', 'predicted_runs', 'predicted_wickets')
    search_fields = ('player__name', 'prediction__match__match_number')
    list_filter = ('prediction__match__season',) 