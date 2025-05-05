"""
Serializers for IPL prediction system models.
"""

from rest_framework import serializers
from .models import (
    Team, Player, Match, PlayerPerformance,
    Prediction, PlayerPrediction
)

class TeamSerializer(serializers.ModelSerializer):
    """Serializer for Team model."""
    class Meta:
        model = Team
        fields = '__all__'

class PlayerSerializer(serializers.ModelSerializer):
    """Serializer for Player model."""
    team = TeamSerializer(read_only=True)
    team_id = serializers.PrimaryKeyRelatedField(
        queryset=Team.objects.all(),
        source='team',
        write_only=True
    )

    class Meta:
        model = Player
        fields = '__all__'

class PlayerPerformanceSerializer(serializers.ModelSerializer):
    """Serializer for PlayerPerformance model."""
    player = PlayerSerializer(read_only=True)
    player_id = serializers.PrimaryKeyRelatedField(
        queryset=Player.objects.all(),
        source='player',
        write_only=True
    )
    team = TeamSerializer(read_only=True)
    team_id = serializers.PrimaryKeyRelatedField(
        queryset=Team.objects.all(),
        source='team',
        write_only=True
    )

    class Meta:
        model = PlayerPerformance
        fields = '__all__'

class MatchSerializer(serializers.ModelSerializer):
    """Serializer for Match model."""
    team1 = TeamSerializer(read_only=True)
    team1_id = serializers.PrimaryKeyRelatedField(
        queryset=Team.objects.all(),
        source='team1',
        write_only=True
    )
    team2 = TeamSerializer(read_only=True)
    team2_id = serializers.PrimaryKeyRelatedField(
        queryset=Team.objects.all(),
        source='team2',
        write_only=True
    )
    toss_winner = TeamSerializer(read_only=True)
    toss_winner_id = serializers.PrimaryKeyRelatedField(
        queryset=Team.objects.all(),
        source='toss_winner',
        write_only=True
    )
    winner = TeamSerializer(read_only=True)
    winner_id = serializers.PrimaryKeyRelatedField(
        queryset=Team.objects.all(),
        source='winner',
        write_only=True,
        required=False
    )
    player_of_match = PlayerSerializer(read_only=True)
    player_of_match_id = serializers.PrimaryKeyRelatedField(
        queryset=Player.objects.all(),
        source='player_of_match',
        write_only=True,
        required=False
    )
    player_performances = PlayerPerformanceSerializer(many=True, read_only=True)

    class Meta:
        model = Match
        fields = '__all__'

class PlayerPredictionSerializer(serializers.ModelSerializer):
    """Serializer for PlayerPrediction model."""
    player = PlayerSerializer(read_only=True)
    player_id = serializers.PrimaryKeyRelatedField(
        queryset=Player.objects.all(),
        source='player',
        write_only=True
    )

    class Meta:
        model = PlayerPrediction
        fields = '__all__'

class PredictionSerializer(serializers.ModelSerializer):
    """Serializer for Prediction model."""
    match = MatchSerializer(read_only=True)
    match_id = serializers.PrimaryKeyRelatedField(
        queryset=Match.objects.all(),
        source='match',
        write_only=True
    )
    predicted_winner = TeamSerializer(read_only=True)
    predicted_winner_id = serializers.PrimaryKeyRelatedField(
        queryset=Team.objects.all(),
        source='predicted_winner',
        write_only=True
    )
    player_predictions = PlayerPredictionSerializer(many=True, read_only=True)

    class Meta:
        model = Prediction
        fields = '__all__'

class PredictionRequestSerializer(serializers.Serializer):
    """Serializer for prediction request data."""
    team1_id = serializers.PrimaryKeyRelatedField(queryset=Team.objects.all())
    team2_id = serializers.PrimaryKeyRelatedField(queryset=Team.objects.all())
    venue = serializers.CharField(max_length=100)
    weather_condition = serializers.CharField(max_length=50, required=False)
    pitch_condition = serializers.CharField(max_length=50, required=False)
    include_player_predictions = serializers.BooleanField(default=False) 