"""
Script to clean and process the consolidated IPL dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        """Initialize the data processor."""
        self.consolidated_dir = Path("consolidated_data")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Load environment variables for database connection
        load_dotenv()
        self.db_params = {
            "dbname": os.getenv("DB_NAME", "ipl_data"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", ""),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
        }

    def load_data(self):
        """Load all consolidated datasets."""
        logger.info("Loading consolidated datasets...")

        self.matches = pd.read_csv(self.consolidated_dir / "consolidated_matches.csv")
        self.deliveries = pd.read_csv(
            self.consolidated_dir / "consolidated_deliveries.csv"
        )
        self.batting_stats = pd.read_csv(
            self.consolidated_dir / "consolidated_batting_stats.csv"
        )
        self.bowling_stats = pd.read_csv(
            self.consolidated_dir / "consolidated_bowling_stats.csv"
        )
        self.players = pd.read_csv(self.consolidated_dir / "consolidated_players.csv")

        logger.info("Datasets loaded successfully!")

    def clean_deliveries(self):
        """Clean and process deliveries data."""
        logger.info("Cleaning deliveries data...")

        # Handle missing values
        self.deliveries["player_dismissed"].fillna("Not Out", inplace=True)
        self.deliveries["dismissal_kind"].fillna("Not Out", inplace=True)
        self.deliveries["fielder"].fillna("None", inplace=True)

        # Create additional features
        self.deliveries["is_boundary"] = self.deliveries["batsman_runs"].apply(
            lambda x: 1 if x >= 4 else 0
        )
        self.deliveries["is_six"] = self.deliveries["batsman_runs"].apply(
            lambda x: 1 if x == 6 else 0
        )
        self.deliveries["is_wicket"] = self.deliveries["player_dismissed"].apply(
            lambda x: 0 if x == "Not Out" else 1
        )

        # Calculate total runs for each ball
        self.deliveries["total_runs"] = self.deliveries[
            "batsman_runs"
        ] + self.deliveries["extra_runs"].fillna(0)

        logger.info("Deliveries data cleaned successfully!")

    def clean_matches(self):
        """Clean and process matches data."""
        logger.info("Cleaning matches data...")

        # Print column names for debugging
        logger.info(f"Matches columns: {self.matches.columns.tolist()}")

        # Convert date to datetime with correct format
        self.matches["date"] = pd.to_datetime(self.matches["date"], format="%d-%m-%Y")

        # Handle missing values
        self.matches["venue"].fillna("Unknown", inplace=True)
        self.matches["city"].fillna("Unknown", inplace=True)

        # Encode categorical features
        self.matches["venue_encoded"] = pd.Categorical(self.matches["venue"]).codes
        self.matches["city_encoded"] = pd.Categorical(self.matches["city"]).codes
        self.matches["team1_encoded"] = pd.Categorical(self.matches["team1"]).codes
        self.matches["team2_encoded"] = pd.Categorical(self.matches["team2"]).codes
        self.matches["winner_encoded"] = pd.Categorical(self.matches["winner"]).codes

        # Calculate total runs and wickets for each match
        match_stats = (
            self.deliveries.groupby("match_id")
            .agg({"total_runs": "sum", "is_wicket": "sum"})
            .reset_index()
        )

        match_stats.columns = ["id", "total_runs", "total_wickets"]

        # Merge match statistics with matches data
        self.matches = self.matches.merge(match_stats, on="id", how="left")

        # Calculate team-specific scores
        team_scores = (
            self.deliveries.groupby(["match_id", "batting_team"])
            .agg({"total_runs": "sum", "is_wicket": "sum"})
            .reset_index()
        )

        # Create team1 and team2 scores
        for team_col in ["team1", "team2"]:
            # Get scores for each team
            team_scores_filtered = team_scores[
                team_scores["batting_team"].isin(self.matches[team_col].unique())
            ]

            # Create a mapping of match_id and team to scores
            score_mapping = team_scores_filtered.set_index(
                ["match_id", "batting_team"]
            )[["total_runs", "is_wicket"]]

            # Apply the mapping to create team scores
            self.matches[f"{team_col}_score"] = self.matches.apply(
                lambda row: score_mapping.get((row["id"], row[team_col]), [0, 0])[0],
                axis=1,
            )
            self.matches[f"{team_col}_wickets"] = self.matches.apply(
                lambda row: score_mapping.get((row["id"], row[team_col]), [0, 0])[1],
                axis=1,
            )

        # Calculate win margins
        self.matches["win_by_runs"] = np.where(
            self.matches["winner"] == self.matches["team1"],
            self.matches["team1_score"] - self.matches["team2_score"],
            self.matches["team2_score"] - self.matches["team1_score"],
        )

        self.matches["win_by_wickets"] = np.where(
            self.matches["winner"] == self.matches["team1"],
            10 - self.matches["team1_wickets"],
            10 - self.matches["team2_wickets"],
        )

        logger.info("Matches data cleaned successfully!")

    def create_player_performance_features(self):
        """Create rolling averages for player performance."""
        logger.info("Creating player performance features...")

        # Sort deliveries by match_id and over
        self.deliveries = self.deliveries.sort_values(["match_id", "over", "ball"])

        # Calculate rolling averages for batsmen
        batsman_stats = (
            self.deliveries.groupby("batsman")
            .agg(
                {
                    "batsman_runs": ["sum", "count", "mean"],
                    "is_boundary": "sum",
                    "is_six": "sum",
                }
            )
            .reset_index()
        )

        batsman_stats.columns = [
            "batsman",
            "total_runs",
            "balls_faced",
            "batting_avg",
            "boundaries",
            "sixes",
        ]

        # Calculate rolling averages for bowlers
        bowler_stats = (
            self.deliveries.groupby("bowler")
            .agg({"is_wicket": "sum", "total_runs": "sum", "ball": "count"})
            .reset_index()
        )

        bowler_stats.columns = ["bowler", "wickets", "runs_conceded", "balls_bowled"]
        bowler_stats["bowling_avg"] = (
            bowler_stats["runs_conceded"] / bowler_stats["wickets"]
        )
        bowler_stats["economy_rate"] = bowler_stats["runs_conceded"] / (
            bowler_stats["balls_bowled"] / 6
        )

        # Merge with existing stats
        self.batting_stats = pd.merge(
            self.batting_stats,
            batsman_stats,
            left_on="player",
            right_on="batsman",
            how="left",
        )
        self.bowling_stats = pd.merge(
            self.bowling_stats,
            bowler_stats,
            left_on="player",
            right_on="bowler",
            how="left",
        )

        logger.info("Player performance features created successfully!")

    def save_processed_data(self):
        """Save processed data to files and database."""
        logger.info("Saving processed data...")

        # Save to CSV files
        self.matches.to_csv(self.processed_dir / "processed_matches.csv", index=False)
        self.deliveries.to_csv(
            self.processed_dir / "processed_deliveries.csv", index=False
        )
        self.batting_stats.to_csv(
            self.processed_dir / "processed_batting_stats.csv", index=False
        )
        self.bowling_stats.to_csv(
            self.processed_dir / "processed_bowling_stats.csv", index=False
        )

        # Save to Parquet files (more efficient storage)
        self.matches.to_parquet(self.processed_dir / "processed_matches.parquet")
        self.deliveries.to_parquet(self.processed_dir / "processed_deliveries.parquet")
        self.batting_stats.to_parquet(
            self.processed_dir / "processed_batting_stats.parquet"
        )
        self.bowling_stats.to_parquet(
            self.processed_dir / "processed_bowling_stats.parquet"
        )

        # Save to PostgreSQL
        try:
            engine = create_engine(
                f"postgresql://{self.db_params['user']}:{self.db_params['password']}@"
                f"{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}"
            )

            # Save to database
            self.matches.to_sql("matches", engine, if_exists="replace", index=False)
            self.deliveries.to_sql(
                "deliveries", engine, if_exists="replace", index=False
            )
            self.batting_stats.to_sql(
                "batting_stats", engine, if_exists="replace", index=False
            )
            self.bowling_stats.to_sql(
                "bowling_stats", engine, if_exists="replace", index=False
            )

            logger.info("Data saved to PostgreSQL successfully!")

        except Exception as e:
            logger.error(f"Error saving to PostgreSQL: {str(e)}")

        logger.info("Processed data saved successfully!")

    def process_data(self):
        """Run the complete data processing pipeline."""
        try:
            self.load_data()
            self.clean_deliveries()  # Clean deliveries first to create is_wicket
            self.clean_matches()
            self.create_player_performance_features()
            self.save_processed_data()

            logger.info("Data processing completed successfully!")

        except Exception as e:
            logger.error(f"Error during data processing: {str(e)}")
            raise


if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_data()
