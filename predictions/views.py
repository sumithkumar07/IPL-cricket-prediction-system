"""
Views for IPL prediction system API endpoints.
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from django.core.cache import cache
from .models import (
    Team, Player, Match, PlayerPerformance,
    Prediction, PlayerPrediction
)
from .serializers import (
    TeamSerializer, PlayerSerializer, MatchSerializer,
    PlayerPerformanceSerializer, PredictionSerializer,
    PlayerPredictionSerializer, PredictionRequestSerializer
)
from ml_model.ensemble import IPLEnsembleModel
from ml_model.llm_reasoning import IPLPredictionExplainer
import logging

logger = logging.getLogger(__name__)

class TeamViewSet(viewsets.ModelViewSet):
    """ViewSet for Team model."""
    queryset = Team.objects.all()
    serializer_class = TeamSerializer
    permission_classes = [IsAuthenticated]

class PlayerViewSet(viewsets.ModelViewSet):
    """ViewSet for Player model."""
    queryset = Player.objects.all()
    serializer_class = PlayerSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Filter players by team if team_id is provided."""
        queryset = Player.objects.all()
        team_id = self.request.query_params.get('team_id', None)
        if team_id is not None:
            queryset = queryset.filter(team_id=team_id)
        return queryset

class MatchViewSet(viewsets.ModelViewSet):
    """ViewSet for Match model."""
    queryset = Match.objects.all()
    serializer_class = MatchSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Filter matches by various parameters."""
        queryset = Match.objects.all()
        
        # Filter by season
        season = self.request.query_params.get('season', None)
        if season is not None:
            queryset = queryset.filter(season=season)
        
        # Filter by team
        team_id = self.request.query_params.get('team_id', None)
        if team_id is not None:
            queryset = queryset.filter(team1_id=team_id) | queryset.filter(team2_id=team_id)
        
        # Filter by date range
        start_date = self.request.query_params.get('start_date', None)
        end_date = self.request.query_params.get('end_date', None)
        if start_date and end_date:
            queryset = queryset.filter(date__range=[start_date, end_date])
        
        return queryset

class PlayerPerformanceViewSet(viewsets.ModelViewSet):
    """ViewSet for PlayerPerformance model."""
    queryset = PlayerPerformance.objects.all()
    serializer_class = PlayerPerformanceSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Filter performances by various parameters."""
        queryset = PlayerPerformance.objects.all()
        
        # Filter by player
        player_id = self.request.query_params.get('player_id', None)
        if player_id is not None:
            queryset = queryset.filter(player_id=player_id)
        
        # Filter by match
        match_id = self.request.query_params.get('match_id', None)
        if match_id is not None:
            queryset = queryset.filter(match_id=match_id)
        
        # Filter by team
        team_id = self.request.query_params.get('team_id', None)
        if team_id is not None:
            queryset = queryset.filter(team_id=team_id)
        
        return queryset

class PredictionViewSet(viewsets.ModelViewSet):
    """ViewSet for Prediction model."""
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Filter predictions by various parameters."""
        queryset = Prediction.objects.all()
        
        # Filter by match
        match_id = self.request.query_params.get('match_id', None)
        if match_id is not None:
            queryset = queryset.filter(match_id=match_id)
        
        # Filter by team
        team_id = self.request.query_params.get('team_id', None)
        if team_id is not None:
            queryset = queryset.filter(
                match__team1_id=team_id
            ) | queryset.filter(
                match__team2_id=team_id
            )
        
        return queryset

    @action(detail=False, methods=['post'])
    def predict(self, request):
        """Generate predictions for a match."""
        try:
            # Validate request data
            serializer = PredictionRequestSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            data = serializer.validated_data
            
            # Check cache first
            cache_key = f"prediction_{data['team1_id']}_{data['team2_id']}_{data['venue']}"
            cached_prediction = cache.get(cache_key)
            if cached_prediction:
                return Response(cached_prediction)
            
            # Prepare data for ML model
            match_data = {
                'team1': data['team1_id'].name,
                'team2': data['team2_id'].name,
                'venue': data['venue'],
                'weather_condition': data.get('weather_condition', 'normal'),
                'pitch_condition': data.get('pitch_condition', 'normal')
            }
            
            # Get predictions from ML model
            ensemble_model = IPLEnsembleModel()
            llm_explainer = IPLPredictionExplainer()
            
            # Get winner prediction
            winner_prediction = llm_explainer.predict_winner(match_data)
            
            # Get score predictions
            score_predictions = ensemble_model.predict_scores(match_data)
            
            # Create response
            response_data = {
                'winner_prediction': winner_prediction,
                'score_predictions': score_predictions
            }
            
            # Cache the prediction
            cache.set(cache_key, response_data, timeout=3600)  # Cache for 1 hour
            
            return Response(response_data)
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class PlayerPredictionViewSet(viewsets.ModelViewSet):
    """ViewSet for PlayerPrediction model."""
    queryset = PlayerPrediction.objects.all()
    serializer_class = PlayerPredictionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Filter predictions by various parameters."""
        queryset = PlayerPrediction.objects.all()
        
        # Filter by player
        player_id = self.request.query_params.get('player_id', None)
        if player_id is not None:
            queryset = queryset.filter(player_id=player_id)
        
        # Filter by prediction
        prediction_id = self.request.query_params.get('prediction_id', None)
        if prediction_id is not None:
            queryset = queryset.filter(prediction_id=prediction_id)
        
        return queryset 