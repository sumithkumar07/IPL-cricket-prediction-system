"""
Script to prepare data for IPL match prediction models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_team_stats(df, team_col, score_col, window=10):
    """Calculate rolling statistics for a team."""
    stats = (
        df.groupby(team_col)[score_col]
        .agg([("mean", "mean"), ("std", "std"), ("max", "max"), ("min", "min")])
        .fillna(0)
    )

    # Calculate rolling statistics
    rolling_stats = (
        df.groupby(team_col)[score_col]
        .rolling(window=window, min_periods=1)
        .agg(
            [
                ("rolling_mean", "mean"),
                ("rolling_std", "std"),
                ("rolling_max", "max"),
                ("rolling_min", "min"),
            ]
        )
        .reset_index()
    )

    return stats, rolling_stats


def calculate_head_to_head(df, team1_col, team2_col, winner_col):
    """Calculate head-to-head statistics between teams."""
    h2h = pd.DataFrame()
    for team1 in df[team1_col].unique():
        for team2 in df[team2_col].unique():
            if team1 != team2:
                matches = df[
                    ((df[team1_col] == team1) & (df[team2_col] == team2))
                    | ((df[team1_col] == team2) & (df[team2_col] == team1))
                ]
                if len(matches) > 0:
                    team1_wins = len(matches[matches[winner_col] == team1])
                    team2_wins = len(matches[matches[winner_col] == team2])
                    h2h = pd.concat(
                        [
                            h2h,
                            pd.DataFrame(
                                {
                                    "team1": [team1],
                                    "team2": [team2],
                                    "h2h_matches": [len(matches)],
                                    "team1_h2h_wins": [team1_wins],
                                    "team2_h2h_wins": [team2_wins],
                                    "h2h_win_ratio": [
                                        team1_wins / len(matches)
                                        if len(matches) > 0
                                        else 0.5
                                    ],
                                }
                            ),
                        ]
                    )
    return h2h


def prepare_model_data():
    """Prepare data for model training."""
    try:
        # Load processed data
        logger.info("Loading processed data...")
        matches = pd.read_csv("data/processed/processed_matches.csv")

        # Convert date to datetime
        matches["date"] = pd.to_datetime(matches["date"])

        # Sort by date to maintain temporal order
        matches = matches.sort_values("date")

        # Calculate team statistics
        logger.info("Calculating team statistics...")
        team1_stats, team1_rolling = calculate_team_stats(
            matches, "team1", "team1_score"
        )
        team2_stats, team2_rolling = calculate_team_stats(
            matches, "team2", "team2_score"
        )

        # Calculate head-to-head statistics
        logger.info("Calculating head-to-head statistics...")
        h2h_stats = calculate_head_to_head(matches, "team1", "team2", "winner")

        # Create features for model
        logger.info("Creating features...")

        # Basic match features
        features = [
            "date",
            "team1",
            "team2",
            "venue_encoded",
            "city_encoded",
            "team1_encoded",
            "team2_encoded",
            "winner_encoded",
            "team1_score",
            "team2_score",
        ]

        # Select features
        model_data = matches[features].copy()

        # Add team statistics
        for team in ["team1", "team2"]:
            stats = team1_stats if team == "team1" else team2_stats
            rolling = team1_rolling if team == "team1" else team2_rolling

            model_data[f"{team}_avg_score"] = model_data[team].map(stats["mean"])
            model_data[f"{team}_score_std"] = model_data[team].map(stats["std"])
            model_data[f"{team}_max_score"] = model_data[team].map(stats["max"])
            model_data[f"{team}_min_score"] = model_data[team].map(stats["min"])

            # Add rolling statistics
            rolling = rolling.set_index("team1" if team == "team1" else "team2")
            model_data[f"{team}_rolling_avg"] = model_data[team].map(
                rolling["rolling_mean"]
            )
            model_data[f"{team}_rolling_std"] = model_data[team].map(
                rolling["rolling_std"]
            )

        # Add head-to-head features
        model_data = model_data.merge(
            h2h_stats, on=["team1", "team2"], how="left"
        ).fillna(0)

        # Create binary winner column (1 if team1 wins, 0 if team2 wins)
        model_data["winner"] = (
            model_data["team1_encoded"] == model_data["winner_encoded"]
        ).astype(int)

        # Drop winner_encoded as we now have binary winner
        model_data = model_data.drop("winner_encoded", axis=1)

        # Convert date to numeric features
        model_data["year"] = model_data["date"].dt.year
        model_data["month"] = model_data["date"].dt.month
        model_data["day"] = model_data["date"].dt.day
        model_data = model_data.drop("date", axis=1)

        # Save the prepared data
        logger.info("Saving prepared data...")
        output_dir = Path("data/processed")
        model_data.to_csv(output_dir / "ipl_matches_processed.csv", index=False)

        logger.info("Data preparation completed successfully!")

    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
        raise


if __name__ == "__main__":
    prepare_model_data()
