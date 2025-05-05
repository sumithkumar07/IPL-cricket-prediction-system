"""
LLM integration for IPL match prediction explanations.
"""

import logging
from typing import Dict, Any
from langchain_community.llms import Ollama
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class IPLPredictionExplainer:
    def __init__(self):
        self.models_dir = Path("models")
        self.llm = Ollama(model="llama2")

        # Load the ensemble model and preprocessors
        self.voting_clf = joblib.load(
            self.models_dir / "voting_classifier_winner.joblib"
        )
        self.scaler = joblib.load(self.models_dir / "ensemble_scaler.joblib")
        self.poly = joblib.load(self.models_dir / "ensemble_poly.joblib")

    def prepare_features(self, match_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction."""
        try:
            # Convert match data to DataFrame
            df = pd.DataFrame([match_data])

            # Scale features
            X_scaled = self.scaler.transform(df)

            # Apply polynomial features
            X_poly = self.poly.transform(X_scaled)

            return X_poly

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def predict_winner(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict match winner and generate explanation."""
        try:
            # Prepare features
            X = self.prepare_features(match_data)

            # Make prediction
            prediction = self.voting_clf.predict(X)[0]
            probabilities = self.voting_clf.predict_proba(X)[0]

            # Generate explanation
            explanation = self.generate_explanation(
                match_data, prediction, probabilities
            )

            return {
                "prediction": prediction,
                "probabilities": probabilities.tolist(),
                "explanation": explanation,
            }

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def generate_explanation(
        self, features: Dict[str, Any], prediction: int, probabilities: np.ndarray
    ) -> str:
        """Generate natural language explanation for the prediction."""
        try:
            # Determine winning team
            winning_team = features["team1"] if prediction == 1 else features["team2"]
            losing_team = features["team2"] if prediction == 1 else features["team1"]

            # Format probabilities
            win_prob = probabilities[1] if prediction == 1 else probabilities[0]
            win_prob_percent = win_prob * 100

            # Create detailed prompt
            prompt = f"""
            Explain why {winning_team} is predicted to win against {losing_team} with {win_prob_percent:.1f}% confidence.
            
            Consider the following factors:
            
            1. Team Form:
            - {winning_team} recent form: {features.get('team1_win_ratio_last5', 'N/A') if prediction == 1 else features.get('team2_win_ratio_last5', 'N/A')}
            - {losing_team} recent form: {features.get('team2_win_ratio_last5', 'N/A') if prediction == 1 else features.get('team1_win_ratio_last5', 'N/A')}
            
            2. Key Players:
            - {winning_team} top batsman: {features.get('team1_top_batsman', 'N/A') if prediction == 1 else features.get('team2_top_batsman', 'N/A')}
            - {winning_team} top bowler: {features.get('team1_top_bowler', 'N/A') if prediction == 1 else features.get('team2_top_bowler', 'N/A')}
            
            3. Venue Impact:
            - Venue: {features.get('venue', 'N/A')}
            - {winning_team} venue record: {features.get('team1_venue_win_rate', 'N/A') if prediction == 1 else features.get('team2_venue_win_rate', 'N/A')}
            
            4. Head-to-Head:
            - {winning_team} vs {losing_team} record: {features.get('team1_h2h_win_rate', 'N/A') if prediction == 1 else features.get('team2_h2h_win_rate', 'N/A')}
            
            5. Weather Impact:
            - Weather conditions: {features.get('weather_condition', 'N/A')}
            - Temperature: {features.get('temperature', 'N/A')}°C
            - Humidity: {features.get('humidity', 'N/A')}%
            - Pitch condition: {features.get('pitch_condition', 'N/A')}
            
            6. Player Availability:
            - {winning_team} key players available: {features.get('team1_key_players_available', 'N/A') if prediction == 1 else features.get('team2_key_players_available', 'N/A')}
            - {losing_team} key players available: {features.get('team2_key_players_available', 'N/A') if prediction == 1 else features.get('team1_key_players_available', 'N/A')}
            
            Provide a detailed analysis considering these factors and explain why {winning_team} is favored to win.
            Also discuss how weather conditions and player availability might impact the match outcome.
            """

            # Generate explanation using LLM
            explanation = self.llm(prompt)

            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise


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
        f"Predicted Winner: {match_data['team1'] if result['prediction'] == 1 else match_data['team2']}"
    )
    print(
        f"Win Probability: {result['probabilities'][1] if result['prediction'] == 1 else result['probabilities'][0]:.2%}"
    )
    print("\nExplanation:")
    print(result["explanation"])
