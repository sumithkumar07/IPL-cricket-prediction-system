"""
LLM integration for IPL prediction explanations.
"""

import logging
from typing import Dict, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler
from langchain_ollama import OllamaLLM
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class IPLPredictionExplainer:
    """Class for generating predictions and explanations using LLM."""
    
    def __init__(self):
        """Initialize the prediction explainer."""
        logger.info("Initializing prediction explainer...")
        self.llm = OllamaLLM(model="llama2:7b-chat")
        self.scaler = StandardScaler()
        self.feature_names = [
            'avg_score_diff', 'city_encoded', 'day', 'dayofweek', 'is_weekend',
            'month', 'team1_avg_score', 'team1_win_rate', 'team2_avg_score',
            'team2_win_rate', 'venue_encoded', 'year'
        ]
        
    def prepare_features(self, match_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for prediction."""
        try:
            # Convert match data to DataFrame
            df = pd.DataFrame([match_data])
            
            # Add required features
            df['day'] = pd.to_datetime(df['date']).dt.day
            df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['year'] = pd.to_datetime(df['date']).dt.year
            
            # Calculate team statistics
            df['team1_avg_score'] = df['team1_avg_score'].fillna(150)  # Default average score
            df['team2_avg_score'] = df['team2_avg_score'].fillna(150)
            df['team1_win_rate'] = df['team1_win_rate'].fillna(0.5)  # Default win rate
            df['team2_win_rate'] = df['team2_win_rate'].fillna(0.5)
            
            # Calculate score difference
            df['avg_score_diff'] = df['team1_avg_score'] - df['team2_avg_score']
            
            # Encode categorical features
            df['city_encoded'] = pd.factorize(df['venue'])[0]
            df['venue_encoded'] = pd.factorize(df['venue'])[0]
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0  # Default value for missing features
            
            # Select only the required features
            X = df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def predict_winner(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction and explanation for match winner."""
        try:
            # Prepare features
            X = self.prepare_features(match_data)
            
            # Calculate confidence using sigmoid function
            score = X[0][0]  # Get the prediction score
            confidence = 1 / (1 + np.exp(-abs(score)))  # Convert to probability using sigmoid
            
            # Generate prediction
            prediction = {
                'winner': 'Team1' if score > 0 else 'Team2',
                'confidence': confidence,
                'explanation': self.generate_explanation(match_data, score > 0)
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def generate_explanation(self, match_data: Dict[str, Any], is_team1_winner: bool) -> str:
        """Generate natural language explanation for the prediction."""
        try:
            prompt = f"""
            Based on the following match data, explain why {'Team1' if is_team1_winner else 'Team2'} is predicted to win:
            
            Match Details:
            - Teams: {match_data['team1']} vs {match_data['team2']}
            - Venue: {match_data['venue']}
            - Date: {match_data['date']}
            - Team1 Stats: Win Rate={match_data.get('team1_win_rate', 0.5)}, Avg Score={match_data.get('team1_avg_score', 150)}
            - Team2 Stats: Win Rate={match_data.get('team2_win_rate', 0.5)}, Avg Score={match_data.get('team2_avg_score', 150)}
            - Weather: {match_data.get('weather_condition', 'normal')}
            - Pitch: {match_data.get('pitch_condition', 'normal')}
            
            Please provide a detailed explanation considering:
            1. Team performance history
            2. Venue advantage
            3. Weather and pitch conditions
            4. Recent form
            5. Key player availability
            """
            
            explanation = self.llm.invoke(prompt)
            return explanation.strip()
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Unable to generate explanation due to an error."


# Example usage
if __name__ == "__main__":
    # Example match data
    match_data = {
        "team1": "Mumbai Indians",
        "team2": "Chennai Super Kings",
        "venue": "Wankhede Stadium",
        "team1_win_ratio_last5": 0.6,
        "team2_win_ratio_last5": 0.4,
        "team1_top_batsman": "Rohit Sharma",
        "team2_top_batsman": "MS Dhoni",
        "team1_top_bowler": "Jasprit Bumrah",
        "team2_top_bowler": "Deepak Chahar",
        "team1_venue_win_rate": 0.7,
        "team2_venue_win_rate": 0.5,
        "team1_h2h_win_rate": 0.55,
        "team2_h2h_win_rate": 0.45,
        "weather_condition": "Clear",
        "temperature": 28,
        "humidity": 65,
        "pitch_condition": "Batting friendly",
        "team1_key_players_available": "All available",
        "team2_key_players_available": "All available",
    }

    # Initialize explainer
    explainer = IPLPredictionExplainer()

    # Get prediction and explanation
    result = explainer.predict_winner(match_data)

    # Print results
    print("\nPrediction Results:")
    print(
        f"Predicted Winner: {match_data['team1'] if result['winner'] == 'Team1' else match_data['team2']}"
    )
    print(
        f"Win Probability: {result['confidence']:.2%}"
    )
    print("\nExplanation:")
    print(result["explanation"])
