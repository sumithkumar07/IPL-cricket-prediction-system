"""
Database models for IPL prediction system.
"""

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class Team(models.Model):
    """Model representing an IPL team."""
    name = models.CharField(max_length=100, unique=True)
    short_name = models.CharField(max_length=10, unique=True)
    founded_year = models.IntegerField()
    home_ground = models.CharField(max_length=100)
    total_matches = models.IntegerField(default=0)
    total_wins = models.IntegerField(default=0)
    total_losses = models.IntegerField(default=0)
    win_percentage = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']

class Player(models.Model):
    """Model representing an IPL player."""
    name = models.CharField(max_length=100)
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='players')
    role = models.CharField(max_length=50)  # Batsman, Bowler, All-rounder
    nationality = models.CharField(max_length=50)
    date_of_birth = models.DateField()
    matches_played = models.IntegerField(default=0)
    total_runs = models.IntegerField(default=0)
    total_wickets = models.IntegerField(default=0)
    batting_average = models.FloatField(default=0.0)
    bowling_average = models.FloatField(default=0.0)
    strike_rate = models.FloatField(default=0.0)
    economy_rate = models.FloatField(default=0.0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.team.short_name})"

    class Meta:
        ordering = ['name']

class Match(models.Model):
    """Model representing an IPL match."""
    match_number = models.IntegerField()
    season = models.IntegerField()
    date = models.DateField()
    venue = models.CharField(max_length=100)
    team1 = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='home_matches')
    team2 = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='away_matches')
    toss_winner = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='toss_wins')
    toss_decision = models.CharField(max_length=10)  # Bat or Field
    winner = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='match_wins', null=True, blank=True)
    team1_score = models.IntegerField(null=True, blank=True)
    team2_score = models.IntegerField(null=True, blank=True)
    team1_wickets = models.IntegerField(null=True, blank=True)
    team2_wickets = models.IntegerField(null=True, blank=True)
    team1_overs = models.FloatField(null=True, blank=True)
    team2_overs = models.FloatField(null=True, blank=True)
    player_of_match = models.ForeignKey(Player, on_delete=models.CASCADE, null=True, blank=True)
    weather_condition = models.CharField(max_length=50, null=True, blank=True)
    pitch_condition = models.CharField(max_length=50, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Match {self.match_number} - {self.team1.short_name} vs {self.team2.short_name}"

    class Meta:
        ordering = ['-date', 'match_number']

class PlayerPerformance(models.Model):
    """Model representing a player's performance in a match."""
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name='player_performances')
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name='performances')
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='player_performances')
    runs_scored = models.IntegerField(default=0)
    balls_faced = models.IntegerField(default=0)
    fours = models.IntegerField(default=0)
    sixes = models.IntegerField(default=0)
    wickets_taken = models.IntegerField(default=0)
    overs_bowled = models.FloatField(default=0.0)
    runs_conceded = models.IntegerField(default=0)
    maidens = models.IntegerField(default=0)
    economy_rate = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.player.name} in {self.match}"

    class Meta:
        ordering = ['-match__date', 'player__name']

class Prediction(models.Model):
    """Model representing match predictions."""
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name='predictions')
    predicted_winner = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='predicted_wins')
    predicted_team1_score = models.IntegerField()
    predicted_team2_score = models.IntegerField()
    confidence = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    model_version = models.CharField(max_length=50)
    weather_impact = models.FloatField(null=True, blank=True)
    pitch_impact = models.FloatField(null=True, blank=True)
    key_factors = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Prediction for {self.match}"

    class Meta:
        ordering = ['-match__date', '-confidence']

class PlayerPrediction(models.Model):
    """Model representing player performance predictions."""
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE, related_name='player_predictions')
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name='predictions')
    predicted_runs = models.IntegerField()
    predicted_wickets = models.IntegerField()
    predicted_economy = models.FloatField()
    confidence = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    key_factors = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Player prediction for {self.player.name} in {self.prediction.match}"

    class Meta:
        ordering = ['-prediction__match__date', '-confidence'] 