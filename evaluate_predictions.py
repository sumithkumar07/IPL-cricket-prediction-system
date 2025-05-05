"""
Script to evaluate model performance on specific IPL prediction tasks.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
from tensorflow.keras.models import load_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.models_dir = Path("models")
        self.data_dir = Path("data")
        self.features_dir = self.data_dir / "features"
        
        # Define feature columns
        self.feature_columns = [
            # Team momentum features
            'team1_win_ratio_last3', 'team1_win_ratio_last5', 'team1_win_ratio_last10',
            'team2_win_ratio_last3', 'team2_win_ratio_last5', 'team2_win_ratio_last10',
            'team1_run_rate_last3', 'team1_run_rate_last5', 'team1_run_rate_last10',
            'team2_run_rate_last3', 'team2_run_rate_last5', 'team2_run_rate_last10',
            'team1_wicket_rate_last3', 'team1_wicket_rate_last5', 'team1_wicket_rate_last10',
            'team2_wicket_rate_last3', 'team2_wicket_rate_last5', 'team2_wicket_rate_last10',
            
            # Score prediction features
            'team1_powerplay_avg', 'team1_powerplay_std',
            'team2_powerplay_avg', 'team2_powerplay_std',
            'team1_middle_overs_avg', 'team1_middle_overs_std',
            'team2_middle_overs_avg', 'team2_middle_overs_std',
            'team1_death_overs_avg', 'team1_death_overs_std',
            'team2_death_overs_avg', 'team2_death_overs_std',
            'team1_run_rate_acceleration', 'team2_run_rate_acceleration',
            'team1_boundary_percentage', 'team2_boundary_percentage',
            'team1_dot_ball_percentage', 'team2_dot_ball_percentage',
            
            # Player performance features
            'avg_total_runs', 'avg_strike_rate', 'avg_boundaries', 'avg_sixes',
            'avg_form_runs_last3', 'avg_form_runs_last5', 'avg_form_runs_last10',
            'avg_form_strike_rate_last3', 'avg_form_strike_rate_last5', 'avg_form_strike_rate_last10',
            'avg_form_boundaries_last3', 'avg_form_boundaries_last5', 'avg_form_boundaries_last10',
            'avg_form_sixes_last3', 'avg_form_sixes_last5', 'avg_form_sixes_last10',
            'avg_wickets', 'avg_economy_rate', 'avg_dot_ball_percentage',
            'avg_form_wickets_last3', 'avg_form_wickets_last5', 'avg_form_wickets_last10',
            'avg_form_economy_last3', 'avg_form_economy_last5', 'avg_form_economy_last10',
            'avg_form_dot_ball_percentage_last3', 'avg_form_dot_ball_percentage_last5',
            'avg_form_dot_ball_percentage_last10',
            
            # Venue impact features
            'venue_avg_runs', 'venue_std_runs',
            'venue_avg_wickets', 'venue_std_wickets',
            'team1_venue_win_rate', 'team2_venue_win_rate',
            'team1_venue_familiarity', 'team2_venue_familiarity',
            
            # Head-to-head features
            'team1_h2h_win_rate', 'team2_h2h_win_rate',
            
            # Weather and pitch features
            'pitch_condition_numeric', 'weather_impact_numeric'
        ]
        
        # Load models
        self.load_models()
        
        # Load test data
        self.load_test_data()
    
    def load_models(self):
        """Load all trained models."""
        logger.info("Loading trained models...")
        
        # Load winner prediction models
        self.xgb_model = joblib.load(self.models_dir / "xgboost_model.joblib")
        self.rf_model = joblib.load(self.models_dir / "random_forest_model.joblib")
        self.nn_model = load_model(self.models_dir / "neural_network_model.keras")
        self.ensemble_model = joblib.load(self.models_dir / "ensemble_model.joblib")
        
        # Load score prediction models
        self.score_xgb_model = joblib.load(self.models_dir / "score_xgboost_model.joblib")
        self.score_rf_model = joblib.load(self.models_dir / "score_random_forest_model.joblib")
        self.score_nn_model = load_model(self.models_dir / "score_neural_network_model.keras")
        self.score_ensemble_model = joblib.load(self.models_dir / "score_ensemble_model.joblib")
        
        # Load scaler
        self.scaler = joblib.load(self.models_dir / "scaler.joblib")
        
        logger.info("Models loaded successfully!")
    
    def load_test_data(self):
        """Load and prepare test data."""
        logger.info("Loading test data...")
        
        # Load engineered features
        features_file = self.features_dir / "engineered_features.parquet"
        if not features_file.exists():
            features_file = self.features_dir / "engineered_features.csv"
        
        self.test_data = pd.read_parquet(features_file) if features_file.suffix == '.parquet' else pd.read_csv(features_file)
        
        # Use the last 20% of matches as test data
        test_size = int(len(self.test_data) * 0.2)
        self.test_data = self.test_data.tail(test_size).reset_index(drop=True)
        
        # Check for missing columns
        missing_columns = [col for col in self.feature_columns if col not in self.test_data.columns]
        if missing_columns:
            logger.warning(f"Missing columns in test data: {missing_columns}")
            for col in missing_columns:
                self.test_data[col] = 0
        
        # Check for NaN values before filling
        nan_columns = self.test_data[self.feature_columns].columns[self.test_data[self.feature_columns].isna().any()].tolist()
        if nan_columns:
            logger.warning(f"Columns with NaN values: {nan_columns}")
            logger.warning("NaN value counts:")
            for col in nan_columns:
                nan_count = self.test_data[col].isna().sum()
                logger.warning(f"{col}: {nan_count} NaN values")
        
        # Fill NaN values with mean or 0 if all values are NaN
        for col in self.feature_columns:
            if self.test_data[col].isna().all():
                logger.warning(f"Column {col} has all NaN values. Setting to 0.")
                self.test_data[col] = 0
            else:
                self.test_data[col] = self.test_data[col].fillna(self.test_data[col].mean())
        
        # Verify no NaN values remain
        remaining_nans = self.test_data[self.feature_columns].isna().sum().sum()
        if remaining_nans > 0:
            logger.error(f"Still have {remaining_nans} NaN values after filling!")
            raise ValueError("Failed to handle all NaN values")
        else:
            logger.info("All NaN values have been handled successfully")
        
        logger.info("Test data loaded and preprocessed successfully!")
    
    def evaluate_match_winner_prediction(self):
        """Evaluate match winner prediction accuracy."""
        logger.info("\nEvaluating Match Winner Prediction:")
        
        # Use all features for prediction
        X = self.test_data[self.feature_columns]
        y_true = (self.test_data['winner'] == self.test_data['team1']).astype(int)
        
        # Verify no NaN values in features
        if X.isna().any().any():
            logger.error("NaN values found in features!")
            nan_cols = X.columns[X.isna().any()].tolist()
            logger.error(f"Columns with NaN values: {nan_cols}")
            raise ValueError("Features contain NaN values")
        
        # Get predictions from each model
        X_scaled = self.scaler.transform(X)
        
        # Verify no NaN values after scaling
        if np.isnan(X_scaled).any():
            logger.error("NaN values found after scaling!")
            raise ValueError("Scaled features contain NaN values")
        
        # Get predictions
        y_pred_xgb = self.xgb_model.predict(X_scaled)
        y_pred_rf = self.rf_model.predict(X_scaled)
        y_pred_nn = self.nn_model.predict(X_scaled).flatten()  # Flatten the output
        y_pred_nn = (y_pred_nn > 0.5).astype(int)  # Convert to binary predictions
        y_pred_ensemble = self.ensemble_model.predict(X_scaled)
        
        # Calculate metrics for each model
        models = {
            'XGBoost': y_pred_xgb,
            'Random Forest': y_pred_rf,
            'Neural Network': y_pred_nn,
            'Ensemble': y_pred_ensemble
        }
        
        for name, predictions in models.items():
            accuracy = accuracy_score(y_true, predictions)
            report = classification_report(y_true, predictions)
            logger.info(f"\n{name} Results:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Classification Report:\n{report}")
    
    def evaluate_score_prediction(self):
        """Evaluate team score prediction accuracy."""
        logger.info("\nEvaluating Team Score Prediction:")
        
        # Use all features for prediction
        X = self.test_data[self.feature_columns]
        y_true_team1 = self.test_data['team1_score']
        y_true_team2 = self.test_data['team2_score']
        
        # Make predictions
        X_scaled = self.scaler.transform(X)
        y_pred_team1 = self.score_ensemble_model.predict(X_scaled)
        y_pred_team2 = self.score_ensemble_model.predict(X_scaled)
        
        # Calculate metrics
        mae_team1 = mean_absolute_error(y_true_team1, y_pred_team1)
        rmse_team1 = np.sqrt(mean_squared_error(y_true_team1, y_pred_team1))
        r2_team1 = r2_score(y_true_team1, y_pred_team1)
        
        mae_team2 = mean_absolute_error(y_true_team2, y_pred_team2)
        rmse_team2 = np.sqrt(mean_squared_error(y_true_team2, y_pred_team2))
        r2_team2 = r2_score(y_true_team2, y_pred_team2)
        
        logger.info(f"Team 1 Score Prediction:")
        logger.info(f"MAE: {mae_team1:.2f} runs")
        logger.info(f"RMSE: {rmse_team1:.2f} runs")
        logger.info(f"R² Score: {r2_team1:.4f}")
        
        logger.info(f"\nTeam 2 Score Prediction:")
        logger.info(f"MAE: {mae_team2:.2f} runs")
        logger.info(f"RMSE: {rmse_team2:.2f} runs")
        logger.info(f"R² Score: {r2_team2:.4f}")
    
    def evaluate_player_performance(self):
        """Evaluate player performance prediction accuracy."""
        logger.info("\nEvaluating Player Performance Prediction:")
        
        # Use all features for prediction
        X = self.test_data[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Evaluate batting metrics
        batting_metrics = ['avg_total_runs', 'avg_strike_rate', 'avg_boundaries']
        for metric in batting_metrics:
            y_true = self.test_data[metric]
            y_pred = self.score_ensemble_model.predict(X_scaled)
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            logger.info(f"\n{metric} Prediction:")
            logger.info(f"MAE: {mae:.2f}")
            logger.info(f"RMSE: {rmse:.2f}")
            logger.info(f"R² Score: {r2:.4f}")
        
        # Evaluate bowling metrics
        bowling_metrics = ['avg_wickets', 'avg_economy_rate', 'avg_dot_ball_percentage']
        for metric in bowling_metrics:
            y_true = self.test_data[metric]
            y_pred = self.score_ensemble_model.predict(X_scaled)
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            logger.info(f"\n{metric} Prediction:")
            logger.info(f"MAE: {mae:.2f}")
            logger.info(f"RMSE: {rmse:.2f}")
            logger.info(f"R² Score: {r2:.4f}")
    
    def evaluate_performance_trends(self):
        """Evaluate prediction accuracy of performance trends."""
        logger.info("\nEvaluating Performance Trends Prediction:")
        
        # Use all features for prediction
        X = self.test_data[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Evaluate trend predictions
        for metric in ['win_ratio', 'run_rate', 'wicket_rate']:
            for n_matches in [3, 5]:
                for team in ['team1', 'team2']:
                    feature = f'{team}_{metric}_last{n_matches}'
                    y_true = self.test_data[feature]
                    y_pred = self.score_ensemble_model.predict(X_scaled)
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2 = r2_score(y_true, y_pred)
                    
                    logger.info(f"\n{feature} Prediction:")
                    logger.info(f"MAE: {mae:.2f}")
                    logger.info(f"RMSE: {rmse:.2f}")
                    logger.info(f"R² Score: {r2:.4f}")
    
    def evaluate_all(self):
        """Run all evaluations."""
        try:
            self.evaluate_match_winner_prediction()
            self.evaluate_score_prediction()
            self.evaluate_player_performance()
            self.evaluate_performance_trends()
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate_all() 