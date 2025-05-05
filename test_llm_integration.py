"""
Test script for LLM integration with IPL prediction model.
"""

import logging
from ml_model.llm_reasoning import IPLPredictionExplainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_prediction():
    """Test the prediction and explanation generation."""
    try:
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
        logger.info("Initializing prediction explainer...")
        explainer = IPLPredictionExplainer()

        # Get prediction and explanation
        logger.info("Generating prediction and explanation...")
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

    except Exception as e:
        logger.error(f"Error in test prediction: {str(e)}")
        raise


if __name__ == "__main__":
    test_prediction()
