"""
Ensemble model for IPL match prediction using XGBoost, Random Forest, and Neural Network.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class IPLEnsembleModel:
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        
    def load_data(self):
        """Load and preprocess the data."""
        try:
            # Load the data
            df = pd.read_csv('data/processed/ipl_matches_processed.csv')
            
            # Create date features
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
            df['dayofweek'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            
            # Label encode categorical variables
            categorical_cols = ['team1', 'team2', 'venue', 'city']
            for col in categorical_cols:
                if col in df.columns:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            
            # Calculate team statistics
            df = self.add_team_stats(df)
            
            # Create interaction features
            df = self.create_interaction_features(df)
            
            # Add venue and city statistics if available
            if 'venue' in df.columns and 'city' in df.columns:
                df = self.add_venue_stats(df)
                df = self.add_city_stats(df)
            
            # Prepare features and targets
            feature_cols = [col for col in df.columns if col.endswith('_encoded') or 
                          col in ['year', 'month', 'day', 'dayofweek', 'is_weekend'] or
                          col.endswith('_rate') or col.endswith('_score') or 
                          col.endswith('_diff') or col.endswith('_streak')]
            
            X = df[feature_cols]
            y_winner = df['winner']
            y_score = df[['team1_score', 'team2_score']]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Add polynomial features
            X_poly = self.poly.fit_transform(X_scaled)
            
            # Split the data
            X_train, X_test, y_winner_train, y_winner_test, y_score_train, y_score_test = train_test_split(
                X_poly, y_winner, y_score, test_size=0.2, random_state=42
            )
            
            return X_train, X_test, y_winner_train, y_winner_test, y_score_train, y_score_test
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def add_team_stats(self, df):
        """Add team statistics features."""
        try:
            # Initialize new columns
            df['team1_win_rate'] = 0.0
            df['team2_win_rate'] = 0.0
            df['team1_avg_score'] = 0.0
            df['team2_avg_score'] = 0.0
            df['team1_score_std'] = 0.0
            df['team2_score_std'] = 0.0
            df['team1_win_streak'] = 0
            df['team2_win_streak'] = 0
            
            # Calculate rolling statistics
            for idx, row in df.iterrows():
                prev_matches = df.loc[:idx-1]
                
                if len(prev_matches) > 0:
                    # Team 1 stats
                    team1_matches = prev_matches[
                        (prev_matches['team1'] == row['team1']) | 
                        (prev_matches['team2'] == row['team1'])
                    ].tail(10)  # Consider last 10 matches
                    
                    if len(team1_matches) > 0:
                        # Win rate
                        team1_wins = sum(
                            (team1_matches['team1'] == row['team1']) & (team1_matches['winner'] == 1) |
                            (team1_matches['team2'] == row['team1']) & (team1_matches['winner'] == 0)
                        )
                        df.at[idx, 'team1_win_rate'] = team1_wins / len(team1_matches)
                        
                        # Score statistics
                        team1_scores = []
                        for _, match in team1_matches.iterrows():
                            if match['team1'] == row['team1']:
                                team1_scores.append(match['team1_score'])
                            else:
                                team1_scores.append(match['team2_score'])
                        
                        if team1_scores:
                            df.at[idx, 'team1_avg_score'] = np.mean(team1_scores)
                            df.at[idx, 'team1_score_std'] = np.std(team1_scores)
                        
                        # Win streak
                        streak = 0
                        for _, match in team1_matches.iloc[::-1].iterrows():
                            if (match['team1'] == row['team1'] and match['winner'] == 1) or \
                               (match['team2'] == row['team1'] and match['winner'] == 0):
                                streak += 1
                            else:
                                break
                        df.at[idx, 'team1_win_streak'] = streak
                    
                    # Team 2 stats
                    team2_matches = prev_matches[
                        (prev_matches['team1'] == row['team2']) | 
                        (prev_matches['team2'] == row['team2'])
                    ].tail(10)
                    
                    if len(team2_matches) > 0:
                        # Win rate
                        team2_wins = sum(
                            (team2_matches['team1'] == row['team2']) & (team2_matches['winner'] == 1) |
                            (team2_matches['team2'] == row['team2']) & (team2_matches['winner'] == 0)
                        )
                        df.at[idx, 'team2_win_rate'] = team2_wins / len(team2_matches)
                        
                        # Score statistics
                        team2_scores = []
                        for _, match in team2_matches.iterrows():
                            if match['team1'] == row['team2']:
                                team2_scores.append(match['team1_score'])
                            else:
                                team2_scores.append(match['team2_score'])
                        
                        if team2_scores:
                            df.at[idx, 'team2_avg_score'] = np.mean(team2_scores)
                            df.at[idx, 'team2_score_std'] = np.std(team2_scores)
                        
                        # Win streak
                        streak = 0
                        for _, match in team2_matches.iloc[::-1].iterrows():
                            if (match['team1'] == row['team2'] and match['winner'] == 1) or \
                               (match['team2'] == row['team2'] and match['winner'] == 0):
                                streak += 1
                            else:
                                break
                        df.at[idx, 'team2_win_streak'] = streak
            
            return df
            
        except Exception as e:
            logger.error(f"Error in add_team_stats: {str(e)}")
            return df
            
    def create_interaction_features(self, df):
        """Create interaction features between team statistics."""
        try:
            # Differences between team stats
            df['win_rate_diff'] = df['team1_win_rate'] - df['team2_win_rate']
            df['avg_score_diff'] = df['team1_avg_score'] - df['team2_avg_score']
            df['score_std_diff'] = df['team1_score_std'] - df['team2_score_std']
            df['win_streak_diff'] = df['team1_win_streak'] - df['team2_win_streak']
            
            # Ratios (avoiding division by zero)
            df['win_rate_ratio'] = df['team1_win_rate'] / (df['team2_win_rate'] + 1e-6)
            df['avg_score_ratio'] = df['team1_avg_score'] / (df['team2_avg_score'] + 1e-6)
            
            # Products
            df['win_rate_product'] = df['team1_win_rate'] * df['team2_win_rate']
            df['avg_score_product'] = df['team1_avg_score'] * df['team2_avg_score']
            df['win_streak_product'] = df['team1_win_streak'] * df['team2_win_streak']
            
            # Squared terms
            df['team1_win_rate^2'] = df['team1_win_rate'] ** 2
            df['team2_win_rate^2'] = df['team2_win_rate'] ** 2
            df['team1_win_streak^2'] = df['team1_win_streak'] ** 2
            df['team2_win_streak^2'] = df['team2_win_streak'] ** 2
            
            # Cross terms
            for stat1 in ['win_rate', 'avg_score', 'win_streak']:
                for stat2 in ['win_rate_diff', 'avg_score_diff', 'win_streak_diff']:
                    if stat1 != stat2.split('_')[0]:
                        df[f'team1_{stat1}_{stat2}'] = df[f'team1_{stat1}'] * df[stat2]
                        df[f'team2_{stat1}_{stat2}'] = df[f'team2_{stat1}'] * df[stat2]
            
            return df
            
        except Exception as e:
            logger.error(f"Error in create_interaction_features: {str(e)}")
            return df
            
    def add_venue_stats(self, df):
        """Add venue-based statistics."""
        try:
            # Initialize venue statistics
            df['venue_win_rate'] = 0.0
            df['venue_avg_score'] = 0.0
            df['venue_matches_played'] = 0
            
            # Calculate venue statistics
            venue_stats = {}
            for idx, row in df.iterrows():
                venue = row['venue']
                prev_matches = df.loc[:idx-1]
                venue_matches = prev_matches[prev_matches['venue'] == venue]
                
                if len(venue_matches) > 0:
                    venue_stats[venue] = {
                        'win_rate': venue_matches['winner'].mean(),
                        'avg_score': (venue_matches['team1_score'].mean() + venue_matches['team2_score'].mean()) / 2,
                        'matches_played': len(venue_matches)
                    }
                    
                    df.at[idx, 'venue_win_rate'] = venue_stats[venue]['win_rate']
                    df.at[idx, 'venue_avg_score'] = venue_stats[venue]['avg_score']
                    df.at[idx, 'venue_matches_played'] = venue_stats[venue]['matches_played']
            
            return df
            
        except Exception as e:
            logger.error(f"Error in add_venue_stats: {str(e)}")
            return df
            
    def add_city_stats(self, df):
        """Add city-based statistics."""
        try:
            # Initialize city statistics
            df['city_win_rate'] = 0.0
            df['city_avg_score'] = 0.0
            df['city_matches_played'] = 0
            
            # Calculate city statistics
            city_stats = {}
            for idx, row in df.iterrows():
                city = row['city']
                prev_matches = df.loc[:idx-1]
                city_matches = prev_matches[prev_matches['city'] == city]
                
                if len(city_matches) > 0:
                    city_stats[city] = {
                        'win_rate': city_matches['winner'].mean(),
                        'avg_score': (city_matches['team1_score'].mean() + city_matches['team2_score'].mean()) / 2,
                        'matches_played': len(city_matches)
                    }
                    
                    df.at[idx, 'city_win_rate'] = city_stats[city]['win_rate']
                    df.at[idx, 'city_avg_score'] = city_stats[city]['avg_score']
                    df.at[idx, 'city_matches_played'] = city_stats[city]['matches_played']
            
            return df
            
        except Exception as e:
            logger.error(f"Error in add_city_stats: {str(e)}")
            return df
            
    def train_winner_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate models for winner prediction."""
        try:
            results = {}
            
            # XGBoost hyperparameter tuning
            xgb_param_grid = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=xgb_param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_xgb_params = grid_search.best_params_
            logger.info(f"Best XGBoost parameters: {best_xgb_params}")
            
            # Train XGBoost with best parameters
            xgb_model = xgb.XGBClassifier(
                **best_xgb_params,
                objective='binary:logistic',
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            # Print feature importance analysis
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            
            # Train and evaluate Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            # Train and evaluate Neural Network
            nn_model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
            
            # Train and evaluate XGBoost
            logger.info("Training XGBoost for winner prediction...")
            xgb_model.fit(X_train, y_train)
            cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
            logger.info(f"XGBoost Cross-validation scores: {cv_scores}")
            logger.info(f"XGBoost Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            xgb_pred = xgb_model.predict(X_test)
            xgb_accuracy = accuracy_score(y_test, xgb_pred)
            logger.info(f"XGBoost accuracy: {xgb_accuracy:.4f}")
            logger.info("XGBoost classification report:")
            logger.info(classification_report(y_test, xgb_pred))
            results['xgboost'] = {'accuracy': xgb_accuracy, 'predictions': xgb_pred}
            
            # Train and evaluate Random Forest
            logger.info("Training Random Forest for winner prediction...")
            rf_model.fit(X_train, y_train)
            cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
            logger.info(f"Random Forest Cross-validation scores: {cv_scores}")
            logger.info(f"Random Forest Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            rf_pred = rf_model.predict(X_test)
            rf_accuracy = accuracy_score(y_test, rf_pred)
            logger.info(f"Random Forest accuracy: {rf_accuracy:.4f}")
            logger.info("Random Forest classification report:")
            logger.info(classification_report(y_test, rf_pred))
            results['random_forest'] = {'accuracy': rf_accuracy, 'predictions': rf_pred}
            
            # Train and evaluate Neural Network
            logger.info("Training Neural Network for winner prediction...")
            nn_model.fit(X_train, y_train)
            cv_scores = cross_val_score(nn_model, X_train, y_train, cv=5)
            logger.info(f"Neural Network Cross-validation scores: {cv_scores}")
            logger.info(f"Neural Network Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            nn_pred = nn_model.predict(X_test)
            nn_accuracy = accuracy_score(y_test, nn_pred)
            logger.info(f"Neural Network accuracy: {nn_accuracy:.4f}")
            logger.info("Neural Network classification report:")
            logger.info(classification_report(y_test, nn_pred))
            results['neural_network'] = {'accuracy': nn_accuracy, 'predictions': nn_pred}
            
            # Train Voting Classifier
            logger.info("Training Voting Classifier for winner prediction...")
            voting_clf = VotingClassifier(
                estimators=[
                    ('xgb', xgb_model),
                    ('rf', rf_model),
                    ('nn', nn_model)
                ],
                voting='soft'
            )
            
            voting_clf.fit(X_train, y_train)
            cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=5)
            logger.info(f"Voting Classifier Cross-validation scores: {cv_scores}")
            logger.info(f"Voting Classifier Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            voting_pred = voting_clf.predict(X_test)
            voting_accuracy = accuracy_score(y_test, voting_pred)
            logger.info(f"Voting Classifier accuracy: {voting_accuracy:.4f}")
            logger.info("Voting Classifier classification report:")
            logger.info(classification_report(y_test, voting_pred))
            results['voting'] = {'accuracy': voting_accuracy, 'predictions': voting_pred}
            
            # Save models
            joblib.dump(xgb_model, 'models/xgboost_winner.joblib')
            joblib.dump(rf_model, 'models/random_forest_winner.joblib')
            joblib.dump(nn_model, 'models/neural_network_winner.joblib')
            joblib.dump(voting_clf, 'models/voting_classifier_winner.joblib')
            
            return results
            
        except Exception as e:
            logger.error(f"Error in train_winner_models: {str(e)}")
            raise
            
    def train_score_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate models for score prediction."""
        try:
            results = {}
            
            # Initialize models for team1 score prediction
            xgb_model_team1 = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
            
            rf_model_team1 = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            nn_model_team1 = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
            
            # Initialize models for team2 score prediction
            xgb_model_team2 = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
            
            rf_model_team2 = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            nn_model_team2 = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
            
            # Train and evaluate models for team1 score prediction
            logger.info("Training models for team1 score prediction...")
            
            # XGBoost Team1
            xgb_model_team1.fit(X_train, y_train['team1_score'])
            xgb_pred_team1 = xgb_model_team1.predict(X_test)
            
            # Random Forest Team1
            rf_model_team1.fit(X_train, y_train['team1_score'])
            rf_pred_team1 = rf_model_team1.predict(X_test)
            
            # Neural Network Team1
            nn_model_team1.fit(X_train, y_train['team1_score'])
            nn_pred_team1 = nn_model_team1.predict(X_test)
            
            # Train and evaluate models for team2 score prediction
            logger.info("Training models for team2 score prediction...")
            
            # XGBoost Team2
            xgb_model_team2.fit(X_train, y_train['team2_score'])
            xgb_pred_team2 = xgb_model_team2.predict(X_test)
            
            # Random Forest Team2
            rf_model_team2.fit(X_train, y_train['team2_score'])
            rf_pred_team2 = rf_model_team2.predict(X_test)
            
            # Neural Network Team2
            nn_model_team2.fit(X_train, y_train['team2_score'])
            nn_pred_team2 = nn_model_team2.predict(X_test)
            
            # Calculate metrics for team1 score predictions
            results['team1'] = {
                'xgboost': {
                    'mae': mean_absolute_error(y_test['team1_score'], xgb_pred_team1),
                    'rmse': np.sqrt(mean_squared_error(y_test['team1_score'], xgb_pred_team1)),
                    'r2': r2_score(y_test['team1_score'], xgb_pred_team1)
                },
                'random_forest': {
                    'mae': mean_absolute_error(y_test['team1_score'], rf_pred_team1),
                    'rmse': np.sqrt(mean_squared_error(y_test['team1_score'], rf_pred_team1)),
                    'r2': r2_score(y_test['team1_score'], rf_pred_team1)
                },
                'neural_network': {
                    'mae': mean_absolute_error(y_test['team1_score'], nn_pred_team1),
                    'rmse': np.sqrt(mean_squared_error(y_test['team1_score'], nn_pred_team1)),
                    'r2': r2_score(y_test['team1_score'], nn_pred_team1)
                }
            }
            
            # Calculate metrics for team2 score predictions
            results['team2'] = {
                'xgboost': {
                    'mae': mean_absolute_error(y_test['team2_score'], xgb_pred_team2),
                    'rmse': np.sqrt(mean_squared_error(y_test['team2_score'], xgb_pred_team2)),
                    'r2': r2_score(y_test['team2_score'], xgb_pred_team2)
                },
                'random_forest': {
                    'mae': mean_absolute_error(y_test['team2_score'], rf_pred_team2),
                    'rmse': np.sqrt(mean_squared_error(y_test['team2_score'], rf_pred_team2)),
                    'r2': r2_score(y_test['team2_score'], rf_pred_team2)
                },
                'neural_network': {
                    'mae': mean_absolute_error(y_test['team2_score'], nn_pred_team2),
                    'rmse': np.sqrt(mean_squared_error(y_test['team2_score'], nn_pred_team2)),
                    'r2': r2_score(y_test['team2_score'], nn_pred_team2)
                }
            }
            
            # Log results
            for team in ['team1', 'team2']:
                logger.info(f"\n{team.upper()} Score Prediction Results:")
                for model_name, metrics in results[team].items():
                    logger.info(f"\n{model_name.upper()}:")
                    logger.info(f"MAE: {metrics['mae']:.2f}")
                    logger.info(f"RMSE: {metrics['rmse']:.2f}")
                    logger.info(f"R² Score: {metrics['r2']:.4f}")
            
            # Save models
            joblib.dump(xgb_model_team1, 'models/xgboost_team1_score.joblib')
            joblib.dump(rf_model_team1, 'models/random_forest_team1_score.joblib')
            joblib.dump(nn_model_team1, 'models/neural_network_team1_score.joblib')
            joblib.dump(xgb_model_team2, 'models/xgboost_team2_score.joblib')
            joblib.dump(rf_model_team2, 'models/random_forest_team2_score.joblib')
            joblib.dump(nn_model_team2, 'models/neural_network_team2_score.joblib')
            
            return results
            
        except Exception as e:
            logger.error(f"Error in train_score_models: {str(e)}")
            raise
            
    def train_and_evaluate(self):
        """Train and evaluate all models."""
        try:
            # Load and preprocess data
            X_train, X_test, y_winner_train, y_winner_test, y_score_train, y_score_test = self.load_data()
            
            # Print feature importance analysis
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            xgb_temp = xgb.XGBClassifier().fit(X_train, y_winner_train)
            rf_temp = RandomForestClassifier().fit(X_train, y_winner_train)
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'XGBoost Importance': xgb_temp.feature_importances_,
                'Random Forest Importance': rf_temp.feature_importances_
            })
            importance_df['Average Importance'] = (importance_df['XGBoost Importance'] + 
                                                 importance_df['Random Forest Importance']) / 2
            importance_df = importance_df.sort_values('Average Importance', ascending=False)
            
            logger.info("\nFeature Importance Analysis:")
            logger.info(importance_df.head(10))
            
            # Train and evaluate winner prediction models
            winner_results = self.train_winner_models(X_train, y_winner_train, X_test, y_winner_test)
            
            # Train and evaluate score prediction models
            score_results = self.train_score_models(X_train, y_score_train, X_test, y_score_test)
            
            # Save preprocessors
            joblib.dump(self.scaler, 'models/ensemble_scaler.joblib')
            joblib.dump(self.poly, 'models/ensemble_poly.joblib')
            for name, encoder in self.label_encoders.items():
                joblib.dump(encoder, f'models/ensemble_{name}_encoder.joblib')
            
            return {
                'winner_results': winner_results,
                'score_results': score_results
            }
            
        except Exception as e:
            logger.error(f"Error in train_and_evaluate: {str(e)}")
            raise

if __name__ == "__main__":
    model = IPLEnsembleModel()
    model.train_and_evaluate() 