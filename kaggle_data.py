"""
Module to download and process IPL data from Kaggle.
"""

import os
import logging
import pandas as pd
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleDataCollector:
    def __init__(self, data_dir: str = "data"):
        """Initialize the Kaggle data collector."""
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        for directory in [self.raw_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kaggle API
        self.api = KaggleApi()
        self.api.authenticate()
    
    def download_ipl_data(self) -> bool:
        """Download IPL datasets from Kaggle."""
        try:
            # List of relevant IPL datasets
            datasets = [
                "patrickb1912/ipl-complete-dataset-2008-2020",  # Complete IPL dataset
                "ramjidoolla/ipl-data-set",  # Recent IPL data
                "yash612/ipl-players-batting-and-bowling-stats"  # Player stats
            ]
            
            for dataset in datasets:
                try:
                    self.logger.info(f"Downloading dataset: {dataset}")
                    self.api.dataset_download_files(
                        dataset,
                        path=self.raw_dir,
                        unzip=True
                    )
                except Exception as e:
                    self.logger.error(f"Error downloading dataset {dataset}: {e}")
            
            # Process downloaded data
            return self.process_data()
            
        except Exception as e:
            self.logger.error(f"Error in download_ipl_data: {e}")
            return False
    
    def process_data(self) -> bool:
        """Process downloaded data files."""
        try:
            # Process matches data
            matches_files = list(self.raw_dir.glob("*matches*.csv"))
            if matches_files:
                matches_data = []
                for file in matches_files:
                    try:
                        df = pd.read_csv(file)
                        matches_data.append(df)
                    except Exception as e:
                        self.logger.error(f"Error reading {file}: {e}")
                
                if matches_data:
                    # Combine all matches data
                    combined_matches = pd.concat(matches_data, ignore_index=True)
                    # Remove duplicates
                    combined_matches = combined_matches.drop_duplicates()
                    # Save processed data
                    combined_matches.to_csv(self.processed_dir / "all_matches.csv", index=False)
                    self.logger.info(f"Processed {len(combined_matches)} matches")
            
            # Process player stats
            player_files = list(self.raw_dir.glob("*player*.csv"))
            if player_files:
                player_data = []
                for file in player_files:
                    try:
                        df = pd.read_csv(file)
                        player_data.append(df)
                    except Exception as e:
                        self.logger.error(f"Error reading {file}: {e}")
                
                if player_data:
                    # Combine all player data
                    combined_players = pd.concat(player_data, ignore_index=True)
                    # Remove duplicates
                    combined_players = combined_players.drop_duplicates()
                    # Save processed data
                    combined_players.to_csv(self.processed_dir / "all_players.csv", index=False)
                    self.logger.info(f"Processed {len(combined_players)} player records")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run data collector
    collector = KaggleDataCollector()
    collector.download_ipl_data() 