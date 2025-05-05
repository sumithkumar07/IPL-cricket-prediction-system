"""
Test script for LLM integration.
"""

import logging
from ml_model.llm_reasoning import IPLPredictionExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prediction():
    """Test the prediction system."""
    try:
        # Initialize the prediction explainer
        logger.info("Initializing prediction explainer...")
        explainer = IPLPredictionExplainer()

        # Test data
        match_data = {
            "team1": "Mumbai Indians",
            "team2": "Chennai Super Kings",
            "venue": "Wankhede Stadium",
            "date": "2024-05-05",
            "team1_avg_score": 165,
            "team2_avg_score": 160,
            "team1_win_rate": 0.6,
            "team2_win_rate": 0.55,
            "weather_condition": "normal",
            "pitch_condition": "batting_friendly"
        }

        # Get prediction
        result = explainer.predict_winner(match_data)

        # Print results
        print("\nPrediction Results:")
        print(f"Predicted Winner: {match_data['team1'] if result['winner'] == 'Team1' else match_data['team2']}")
        print(f"Win Probability: {result['confidence']:.2%}")
        print("\nExplanation:")
        print(result['explanation'])

    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        raise

if __name__ == "__main__":
    test_prediction()
