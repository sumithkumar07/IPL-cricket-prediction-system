"""
Django models for storing IPL data.
"""

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class Team(models.Model):
    """Model for storing IPL team information."""
    
    name = models.CharField(max_length=100, unique=True)
    short_name = models.CharField(max_length=10)
    founded_year = models.IntegerField()
    home_ground = models.CharField(max_length=100)
    total_matches = models.IntegerField(default=0)
    wins = models.IntegerField(default=0)
    losses = models.IntegerField(default=0)
    win_percentage = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )
    
    def __str__(self):
        return self.name
        
    class Meta:
        ordering = ['name']

class Player(models.Model):
    """Model for storing IPL player information."""
    
    ROLE_CHOICES = [
        ('BAT', 'Batsman'),
        ('BOWL', 'Bowler'),
        ('ALL', 'All-rounder'),
        ('WK', 'Wicket-keeper')
    ]
    
    name = models.CharField(max_length=100)
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='players')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    nationality = models.CharField(max_length=50)
    date_of_birth = models.DateField()
    batting_stats = models.JSONField(default=dict)
    bowling_stats = models.JSONField(default=dict)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.name} ({self.team.short_name})"
        
    class Meta:
        ordering = ['name']
        unique_together = ['name', 'team']

class Match(models.Model):
    """Model for storing IPL match information."""
    
    match_number = models.CharField(max_length=20)
    season = models.CharField(max_length=10)
    date = models.DateTimeField()
    venue = models.CharField(max_length=100)
    team1 = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name='matches_as_team1'
    )
    team2 = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name='matches_as_team2'
    )
    toss_winner = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name='matches_won_toss'
    )
    winner = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name='matches_won',
        null=True,
        blank=True
    )
    team1_score = models.IntegerField()
    team1_wickets = models.IntegerField()
    team2_score = models.IntegerField()
    team2_wickets = models.IntegerField()
    player_of_match = models.ForeignKey(
        Player,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='player_of_match_awards'
    )
    weather_condition = models.CharField(max_length=50)
    pitch_condition = models.CharField(max_length=50)
    
    def __str__(self):
        return f"{self.team1.short_name} vs {self.team2.short_name} - {self.season} Match {self.match_number}"
        
    class Meta:
        ordering = ['-date']
        unique_together = ['match_number', 'season']

class PlayerPerformance(models.Model):
    """Model for storing player performance in a match."""
    
    match = models.ForeignKey(
        Match,
        on_delete=models.CASCADE,
        related_name='player_performances'
    )
    player = models.ForeignKey(
        Player,
        on_delete=models.CASCADE,
        related_name='performances'
    )
    runs_scored = models.IntegerField(default=0)
    balls_faced = models.IntegerField(default=0)
    fours = models.IntegerField(default=0)
    sixes = models.IntegerField(default=0)
    wickets_taken = models.IntegerField(default=0)
    overs_bowled = models.FloatField(default=0.0)
    runs_conceded = models.IntegerField(default=0)
    maidens = models.IntegerField(default=0)
    catches = models.IntegerField(default=0)
    stumpings = models.IntegerField(default=0)
    
    def __str__(self):
        return f"{self.player.name} in {self.match}"
        
    class Meta:
        ordering = ['-match__date']
        unique_together = ['match', 'player']

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