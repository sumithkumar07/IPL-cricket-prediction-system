"""
Analyze IPL data and generate insights.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class IPLAnalyzer:
    def __init__(self, data_dir: str = "data"):
        """Initialize the IPL data analyzer."""
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"

        # Set up plotting style
        plt.style.use("seaborn")
        sns.set_palette("husl")

    def load_data(self):
        """Load all processed data files."""
        self.matches = pd.read_csv(self.processed_dir / "matches.csv")
        self.teams = pd.read_csv(self.processed_dir / "teams.csv")
        self.batting_stats = pd.read_csv(self.processed_dir / "batting_stats.csv")
        self.batting_perf = pd.read_csv(self.processed_dir / "batting_performance.csv")
        self.bowling_perf = pd.read_csv(self.processed_dir / "bowling_performance.csv")
        self.team_perf = pd.read_csv(self.processed_dir / "team_performance.csv")
        self.innings = pd.read_csv(self.processed_dir / "innings_summary.csv")

    def analyze_team_performance(self):
        """Analyze team performance metrics."""
        # Sort teams by win rate
        team_stats = self.team_perf.sort_values("win_rate", ascending=False)

        # Create bar plot of win rates
        plt.figure(figsize=(12, 6))
        sns.barplot(data=team_stats, x="team", y="win_rate")
        plt.title("Team Win Rates in IPL")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.data_dir / "plots" / "team_win_rates.png")
        plt.close()

        # Print team statistics
        print("\nTeam Performance Summary:")
        print(
            team_stats[["team", "matches_played", "matches_won", "win_rate"]].to_string(
                index=False
            )
        )

    def analyze_batting_performance(self):
        """Analyze batting performance metrics."""
        # Get top batsmen by runs
        top_batsmen = self.batting_perf.sort_values("total_runs", ascending=False).head(
            10
        )

        # Create bar plot of top run scorers
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_batsmen, x="player", y="total_runs")
        plt.title("Top 10 Run Scorers in IPL")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.data_dir / "plots" / "top_run_scorers.png")
        plt.close()

        # Print batting statistics
        print("\nTop 10 Batsmen by Runs:")
        print(
            top_batsmen[
                [
                    "player",
                    "total_runs",
                    "matches_played",
                    "strike_rate",
                    "average_runs_per_match",
                ]
            ].to_string(index=False)
        )

    def analyze_bowling_performance(self):
        """Analyze bowling performance metrics."""
        # Get top bowlers by wickets
        top_bowlers = self.bowling_perf.sort_values("wickets", ascending=False).head(10)

        # Create bar plot of top wicket takers
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_bowlers, x="player", y="wickets")
        plt.title("Top 10 Wicket Takers in IPL")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.data_dir / "plots" / "top_wicket_takers.png")
        plt.close()

        # Print bowling statistics
        print("\nTop 10 Bowlers by Wickets:")
        print(
            top_bowlers[["player", "wickets", "matches_bowled", "economy"]].to_string(
                index=False
            )
        )

    def analyze_match_trends(self):
        """Analyze match trends and statistics."""
        # Calculate average runs per innings
        innings_stats = (
            self.innings.groupby("batting_team")["runs"]
            .agg(["mean", "max", "min"])
            .round(2)
        )
        innings_stats.columns = ["Average Runs", "Highest Score", "Lowest Score"]

        # Create box plot of runs distribution by team
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.innings, x="batting_team", y="runs")
        plt.title("Distribution of Team Scores in IPL")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.data_dir / "plots" / "team_scores_distribution.png")
        plt.close()

        # Print innings statistics
        print("\nTeam Batting Statistics:")
        print(innings_stats.to_string())

    def generate_insights(self):
        """Generate comprehensive insights from the data."""
        # Create plots directory if it doesn't exist
        plots_dir = self.data_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Load data
        self.load_data()

        # Generate analyses
        self.analyze_team_performance()
        self.analyze_batting_performance()
        self.analyze_bowling_performance()
        self.analyze_match_trends()


if __name__ == "__main__":
    # Initialize and run analyzer
    analyzer = IPLAnalyzer()
    analyzer.generate_insights()
