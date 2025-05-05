"""
Script to create enhanced features for IPL match prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self):
        """Initialize the feature engineer."""
        self.data_dir = Path("data/processed")
        self.output_dir = Path("data/features")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize scaler
        self.scaler = StandardScaler()

    def load_data(self):
        """Load processed data."""
        logger.info("Loading processed data...")

        self.matches = pd.read_parquet(self.data_dir / "processed_matches.parquet")
        self.deliveries = pd.read_parquet(
            self.data_dir / "processed_deliveries.parquet"
        )
        self.batting_stats = pd.read_parquet(
            self.data_dir / "processed_batting_stats.parquet"
        )
        self.bowling_stats = pd.read_parquet(
            self.data_dir / "processed_bowling_stats.parquet"
        )

        # Print column names for debugging
        logger.info(f"Matches columns: {self.matches.columns.tolist()}")
        logger.info(f"Deliveries columns: {self.deliveries.columns.tolist()}")

        logger.info("Data loaded successfully!")

    def calculate_team_momentum(self):
        """Calculate enhanced team momentum features."""
        logger.info("Calculating enhanced team momentum features...")

        # Sort matches by date
        self.matches = self.matches.sort_values("date")

        # Calculate rolling statistics for each team
        for team_col in ["team1", "team2"]:
            # Create win indicator
            self.matches[f"{team_col}_won"] = (
                self.matches["winner"] == self.matches[team_col]
            ).astype(int)

            # Win ratio
            self.matches[f"{team_col}_win_ratio_last3"] = self.matches.groupby(
                team_col
            )[f"{team_col}_won"].transform(lambda x: x.rolling(3, min_periods=1).mean())
            self.matches[f"{team_col}_win_ratio_last5"] = self.matches.groupby(
                team_col
            )[f"{team_col}_won"].transform(lambda x: x.rolling(5, min_periods=1).mean())
            self.matches[f"{team_col}_win_ratio_last10"] = self.matches.groupby(
                team_col
            )[f"{team_col}_won"].transform(
                lambda x: x.rolling(10, min_periods=1).mean()
            )

            # Run rate
            self.matches[f"{team_col}_run_rate_last3"] = self.matches.groupby(team_col)[
                f"{team_col}_score"
            ].transform(lambda x: x.rolling(3, min_periods=1).mean())
            self.matches[f"{team_col}_run_rate_last5"] = self.matches.groupby(team_col)[
                f"{team_col}_score"
            ].transform(lambda x: x.rolling(5, min_periods=1).mean())
            self.matches[f"{team_col}_run_rate_last10"] = self.matches.groupby(
                team_col
            )[f"{team_col}_score"].transform(
                lambda x: x.rolling(10, min_periods=1).mean()
            )

            # Wicket rate
            self.matches[f"{team_col}_wicket_rate_last3"] = self.matches.groupby(
                team_col
            )[f"{team_col}_wickets"].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            self.matches[f"{team_col}_wicket_rate_last5"] = self.matches.groupby(
                team_col
            )[f"{team_col}_wickets"].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
            self.matches[f"{team_col}_wicket_rate_last10"] = self.matches.groupby(
                team_col
            )[f"{team_col}_wickets"].transform(
                lambda x: x.rolling(10, min_periods=1).mean()
            )

        logger.info("Enhanced team momentum features created successfully!")

    def calculate_phase_wise_runs(self):
        """Calculate runs scored in different phases of the match."""
        logger.info("Calculating phase-wise runs...")

        # Calculate phase-wise runs for each team
        for team_col in ["team1", "team2"]:
            # Get unique teams
            teams = self.matches[team_col].unique()

            for team in teams:
                # Powerplay (1-6 overs)
                powerplay_runs = (
                    self.deliveries[
                        (self.deliveries["batting_team"] == team)
                        & (self.deliveries["over"] <= 6)
                    ]
                    .groupby("match_id")["total_runs"]
                    .sum()
                )

                # Middle overs (7-15 overs)
                middle_overs_runs = (
                    self.deliveries[
                        (self.deliveries["batting_team"] == team)
                        & (self.deliveries["over"] > 6)
                        & (self.deliveries["over"] <= 15)
                    ]
                    .groupby("match_id")["total_runs"]
                    .sum()
                )

                # Death overs (16-20 overs)
                death_overs_runs = (
                    self.deliveries[
                        (self.deliveries["batting_team"] == team)
                        & (self.deliveries["over"] > 15)
                    ]
                    .groupby("match_id")["total_runs"]
                    .sum()
                )

                # Calculate total balls faced
                total_balls = (
                    self.deliveries[self.deliveries["batting_team"] == team]
                    .groupby("match_id")["ball"]
                    .count()
                )

                # Calculate boundaries and dot balls
                boundaries = (
                    self.deliveries[
                        (self.deliveries["batting_team"] == team)
                        & (self.deliveries["is_boundary"] == 1)
                    ]
                    .groupby("match_id")["is_boundary"]
                    .sum()
                )

                dot_balls = (
                    self.deliveries[
                        (self.deliveries["batting_team"] == team)
                        & (self.deliveries["total_runs"] == 0)
                    ]
                    .groupby("match_id")["ball"]
                    .count()
                )

                # Update matches DataFrame for the current team
                team_matches = self.matches[self.matches[team_col] == team]
                self.matches.loc[
                    team_matches.index, f"{team_col}_powerplay_runs"
                ] = powerplay_runs
                self.matches.loc[
                    team_matches.index, f"{team_col}_middle_overs_runs"
                ] = middle_overs_runs
                self.matches.loc[
                    team_matches.index, f"{team_col}_death_overs_runs"
                ] = death_overs_runs
                self.matches.loc[
                    team_matches.index, f"{team_col}_total_balls"
                ] = total_balls
                self.matches.loc[
                    team_matches.index, f"{team_col}_boundaries"
                ] = boundaries
                self.matches.loc[
                    team_matches.index, f"{team_col}_dot_balls"
                ] = dot_balls

        # Fill NaN values with 0
        phase_cols = [
            "powerplay_runs",
            "middle_overs_runs",
            "death_overs_runs",
            "total_balls",
            "boundaries",
            "dot_balls",
        ]
        for team_col in ["team1", "team2"]:
            for col in phase_cols:
                self.matches[f"{team_col}_{col}"] = self.matches[
                    f"{team_col}_{col}"
                ].fillna(0)

        logger.info("Phase-wise runs calculated successfully!")

    def add_weather_and_pitch_features(self):
        """Add weather and pitch condition features."""
        logger.info("Adding weather and pitch condition features...")

        # Calculate venue statistics
        venue_stats = (
            self.matches.groupby("venue")
            .agg({"total_runs": ["mean", "std"], "total_wickets": ["mean", "std"]})
            .reset_index()
        )

        venue_stats.columns = [
            "venue",
            "venue_avg_runs",
            "venue_std_runs",
            "venue_avg_wickets",
            "venue_std_wickets",
        ]

        # Merge venue statistics with matches
        self.matches = self.matches.merge(venue_stats, on="venue", how="left")

        # Create pitch condition features (encoded as numeric)
        self.matches["pitch_condition_numeric"] = np.where(
            self.matches["venue_avg_runs"] > self.matches["venue_avg_runs"].mean(),
            1,  # batting_friendly
            0,  # bowling_friendly
        )

        # Add weather impact (placeholder - would need actual weather data)
        self.matches["weather_impact_numeric"] = 0  # Neutral impact by default

        logger.info("Weather and pitch features added successfully!")

    def add_enhanced_head_to_head_features(self):
        """Calculate enhanced head-to-head statistics."""
        logger.info("Calculating enhanced head-to-head features...")

        # Calculate head-to-head win rates
        for team_col in ["team1", "team2"]:
            # Calculate wins against each opponent
            opponent_col = "team2" if team_col == "team1" else "team1"

            # Group by team and opponent to calculate win rates
            h2h_stats = (
                self.matches.groupby([team_col, opponent_col])
                .agg(
                    {
                        "id": "count",  # Total matches
                        "winner": lambda x: (x == x.name[0]).sum(),  # Wins for team_col
                    }
                )
                .reset_index()
            )

            # Calculate win rate
            h2h_stats["win_rate"] = h2h_stats["winner"] / h2h_stats["id"]

            # Merge back with matches
            self.matches = self.matches.merge(
                h2h_stats[[team_col, opponent_col, "win_rate"]],
                on=[team_col, opponent_col],
                how="left",
            )

            # Rename the win rate column
            self.matches = self.matches.rename(
                columns={"win_rate": f"{team_col}_h2h_win_rate"}
            )

        logger.info("Enhanced head-to-head features created successfully!")

    def add_player_form_features(self):
        """Add enhanced player form features."""
        logger.info("Adding enhanced player form features...")

        # Sort deliveries by match_id and over
        self.deliveries = self.deliveries.sort_values(["match_id", "over", "ball"])

        # Calculate player-wise statistics from deliveries
        batsman_stats = (
            self.deliveries.groupby(["match_id", "batsman"])
            .agg(
                {
                    "batsman_runs": ["sum", "count", "mean"],
                    "is_boundary": "sum",
                    "is_six": "sum",
                    "ball": "count",
                }
            )
            .reset_index()
        )

        batsman_stats.columns = [
            "match_id",
            "batsman",
            "total_runs",
            "balls_faced",
            "batting_avg",
            "boundaries",
            "sixes",
            "total_balls",
        ]

        # Calculate strike rate
        batsman_stats["strike_rate"] = (
            batsman_stats["total_runs"] / batsman_stats["balls_faced"]
        ) * 100

        bowler_stats = (
            self.deliveries.groupby(["match_id", "bowler"])
            .agg(
                {
                    "is_wicket": "sum",
                    "total_runs": "sum",
                    "ball": "count",
                    "is_boundary": "sum",
                    "is_six": "sum",
                }
            )
            .reset_index()
        )

        bowler_stats.columns = [
            "match_id",
            "bowler",
            "wickets",
            "runs_conceded",
            "balls_bowled",
            "boundaries_conceded",
            "sixes_conceded",
        ]

        # Calculate bowling metrics
        bowler_stats["economy_rate"] = bowler_stats["runs_conceded"] / (
            bowler_stats["balls_bowled"] / 6
        )
        bowler_stats["dot_ball_percentage"] = (
            (
                bowler_stats["balls_bowled"]
                - bowler_stats["boundaries_conceded"]
                - bowler_stats["sixes_conceded"]
            )
            / bowler_stats["balls_bowled"]
        ) * 100

        # Sort by match_id to ensure chronological order
        batsman_stats = batsman_stats.sort_values("match_id")
        bowler_stats = bowler_stats.sort_values("match_id")

        # Calculate rolling averages for batsmen
        for window in [3, 5, 10]:
            # Batting form
            batsman_stats[f"form_runs_last{window}"] = batsman_stats.groupby("batsman")[
                "total_runs"
            ].transform(lambda x: x.rolling(window, min_periods=1).mean())
            batsman_stats[f"form_strike_rate_last{window}"] = batsman_stats.groupby(
                "batsman"
            )["strike_rate"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            batsman_stats[f"form_boundaries_last{window}"] = batsman_stats.groupby(
                "batsman"
            )["boundaries"].transform(lambda x: x.rolling(window, min_periods=1).mean())
            batsman_stats[f"form_sixes_last{window}"] = batsman_stats.groupby(
                "batsman"
            )["sixes"].transform(lambda x: x.rolling(window, min_periods=1).mean())

            # Bowling form
            bowler_stats[f"form_wickets_last{window}"] = bowler_stats.groupby("bowler")[
                "wickets"
            ].transform(lambda x: x.rolling(window, min_periods=1).mean())
            bowler_stats[f"form_economy_last{window}"] = bowler_stats.groupby("bowler")[
                "economy_rate"
            ].transform(lambda x: x.rolling(window, min_periods=1).mean())
            bowler_stats[
                f"form_dot_ball_percentage_last{window}"
            ] = bowler_stats.groupby("bowler")["dot_ball_percentage"].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        # Calculate match-level statistics
        match_batting_stats = (
            batsman_stats.groupby("match_id")
            .agg(
                {
                    "total_runs": "mean",
                    "strike_rate": "mean",
                    "boundaries": "mean",
                    "sixes": "mean",
                    "form_runs_last3": "mean",
                    "form_runs_last5": "mean",
                    "form_runs_last10": "mean",
                    "form_strike_rate_last3": "mean",
                    "form_strike_rate_last5": "mean",
                    "form_strike_rate_last10": "mean",
                    "form_boundaries_last3": "mean",
                    "form_boundaries_last5": "mean",
                    "form_boundaries_last10": "mean",
                    "form_sixes_last3": "mean",
                    "form_sixes_last5": "mean",
                    "form_sixes_last10": "mean",
                }
            )
            .reset_index()
        )

        match_bowling_stats = (
            bowler_stats.groupby("match_id")
            .agg(
                {
                    "wickets": "mean",
                    "economy_rate": "mean",
                    "dot_ball_percentage": "mean",
                    "form_wickets_last3": "mean",
                    "form_wickets_last5": "mean",
                    "form_wickets_last10": "mean",
                    "form_economy_last3": "mean",
                    "form_economy_last5": "mean",
                    "form_economy_last10": "mean",
                    "form_dot_ball_percentage_last3": "mean",
                    "form_dot_ball_percentage_last5": "mean",
                    "form_dot_ball_percentage_last10": "mean",
                }
            )
            .reset_index()
        )

        # Rename columns for clarity
        match_batting_stats.columns = ["match_id"] + [
            f"avg_{col}" for col in match_batting_stats.columns if col != "match_id"
        ]
        match_bowling_stats.columns = ["match_id"] + [
            f"avg_{col}" for col in match_bowling_stats.columns if col != "match_id"
        ]

        # Rename match_id to id for merging
        match_batting_stats = match_batting_stats.rename(columns={"match_id": "id"})
        match_bowling_stats = match_bowling_stats.rename(columns={"match_id": "id"})

        # Merge with matches
        self.matches = self.matches.merge(match_batting_stats, on="id", how="left")
        self.matches = self.matches.merge(match_bowling_stats, on="id", how="left")

        logger.info("Enhanced player form features created successfully!")

    def add_venue_specific_features(self):
        """Add venue-specific performance features."""
        logger.info("Adding venue-specific performance features...")

        # Calculate venue statistics for each team
        for team_col in ["team1", "team2"]:
            venue_team_stats = (
                self.matches.groupby(["venue", team_col])
                .agg(
                    {
                        "winner": lambda x: (x == x.name[1]).mean(),
                        f"{team_col}_score": "mean",
                        f"{team_col}_wickets": "mean",
                    }
                )
                .reset_index()
            )

            venue_team_stats.columns = [
                "venue",
                team_col,
                f"{team_col}_venue_win_rate",
                f"{team_col}_venue_avg_score",
                f"{team_col}_venue_avg_wickets",
            ]

            # Merge with matches
            self.matches = self.matches.merge(
                venue_team_stats, on=["venue", team_col], how="left"
            )

        # Calculate venue familiarity
        for team_col in ["team1", "team2"]:
            self.matches[f"{team_col}_venue_familiarity"] = self.matches.groupby(
                [team_col, "venue"]
            )["id"].transform("count")

        logger.info("Venue-specific performance features added successfully!")

    def add_enhanced_score_prediction_features(self):
        """Add enhanced score prediction features including phase-wise statistics."""
        logger.info("Adding enhanced score prediction features...")

        # Calculate phase-wise statistics for each team
        for team_col in ["team1", "team2"]:
            # Powerplay (1-6 overs)
            self.matches[f"{team_col}_powerplay_avg"] = self.matches.groupby(team_col)[
                f"{team_col}_powerplay_runs"
            ].transform("mean")
            self.matches[f"{team_col}_powerplay_std"] = self.matches.groupby(team_col)[
                f"{team_col}_powerplay_runs"
            ].transform("std")

            # Middle overs (7-15 overs)
            self.matches[f"{team_col}_middle_overs_avg"] = self.matches.groupby(
                team_col
            )[f"{team_col}_middle_overs_runs"].transform("mean")
            self.matches[f"{team_col}_middle_overs_std"] = self.matches.groupby(
                team_col
            )[f"{team_col}_middle_overs_runs"].transform("std")

            # Death overs (16-20 overs)
            self.matches[f"{team_col}_death_overs_avg"] = self.matches.groupby(
                team_col
            )[f"{team_col}_death_overs_runs"].transform("mean")
            self.matches[f"{team_col}_death_overs_std"] = self.matches.groupby(
                team_col
            )[f"{team_col}_death_overs_runs"].transform("std")

            # Calculate run rate acceleration
            self.matches[f"{team_col}_run_rate_acceleration"] = (
                self.matches[f"{team_col}_death_overs_avg"]
                - self.matches[f"{team_col}_powerplay_avg"]
            )

            # Calculate boundary percentage
            self.matches[f"{team_col}_boundary_percentage"] = (
                self.matches[f"{team_col}_boundaries"]
                / self.matches[f"{team_col}_total_balls"]
            ) * 100

            # Calculate dot ball percentage
            self.matches[f"{team_col}_dot_ball_percentage"] = (
                self.matches[f"{team_col}_dot_balls"]
                / self.matches[f"{team_col}_total_balls"]
            ) * 100

        logger.info("Enhanced score prediction features added successfully!")

    def engineer_features(self):
        """Run the complete feature engineering pipeline."""
        try:
            self.load_data()
            self.calculate_team_momentum()
            self.calculate_phase_wise_runs()
            self.add_weather_and_pitch_features()
            self.add_enhanced_head_to_head_features()
            self.add_player_form_features()
            self.add_venue_specific_features()
            self.add_enhanced_score_prediction_features()

            # Scale numerical features
            numerical_cols = self.matches.select_dtypes(include=[np.number]).columns
            self.matches[numerical_cols] = self.scaler.fit_transform(
                self.matches[numerical_cols]
            )

            # Save engineered features
            self.matches.to_csv(
                self.output_dir / "engineered_features.csv", index=False
            )
            self.matches.to_parquet(self.output_dir / "engineered_features.parquet")

            logger.info("Feature engineering completed successfully!")

        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            raise


if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.engineer_features()
