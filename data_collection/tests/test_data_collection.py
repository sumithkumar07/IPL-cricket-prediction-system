"""
Tests for the data collection pipeline.
"""

import unittest
from unittest.mock import patch, MagicMock
from django.test import TestCase
from data_collection.managers.data_collection_manager import DataCollectionManager
from data_collection.scrapers.espncricinfo import ESPNCricinfoScraper
from data_collection.processors.data_processor import DataProcessor
from data_collection.storage.storage_manager import StorageManager

class TestDataCollection(TestCase):
    """Test cases for the data collection pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = DataCollectionManager()
        
    @patch.object(ESPNCricinfoScraper, 'scrape_match_data')
    @patch.object(DataProcessor, 'process_match_data')
    @patch.object(StorageManager, 'store_match_data')
    def test_collect_match_data(self, mock_store, mock_process, mock_scrape):
        """Test collecting match data."""
        # Mock data
        mock_scrape.return_value = {
            'match_number': '1',
            'season': '2023',
            'date': '2023-03-31',
            'venue': 'Wankhede Stadium',
            'teams': ['Mumbai Indians', 'Chennai Super Kings'],
            'toss_winner': 'Mumbai Indians',
            'winner': 'Mumbai Indians',
            'scores': {
                'Mumbai Indians': {'runs': 180, 'wickets': 4},
                'Chennai Super Kings': {'runs': 170, 'wickets': 8}
            },
            'player_of_match': 'Rohit Sharma',
            'weather_condition': 'Clear',
            'pitch_condition': 'Good for batting'
        }
        
        mock_process.return_value = mock_scrape.return_value
        mock_store.return_value = MagicMock()
        
        # Test
        result = self.manager.collect_match_data('http://example.com/match/1')
        self.assertIsNotNone(result)
        self.assertEqual(result['match_number'], '1')
        
        # Verify calls
        mock_scrape.assert_called_once_with('http://example.com/match/1')
        mock_process.assert_called_once_with(mock_scrape.return_value)
        mock_store.assert_called_once_with(mock_process.return_value)
        
    @patch.object(ESPNCricinfoScraper, 'scrape_player_data')
    @patch.object(DataProcessor, 'process_player_data')
    @patch.object(StorageManager, 'store_player_data')
    def test_collect_player_data(self, mock_store, mock_process, mock_scrape):
        """Test collecting player data."""
        # Mock data
        mock_scrape.return_value = {
            'name': 'Virat Kohli',
            'team': 'Royal Challengers Bangalore',
            'role': 'Batsman',
            'nationality': 'Indian',
            'date_of_birth': '1988-11-05',
            'batting_stats': {
                'matches': 200,
                'runs': 6000,
                'average': 40.0,
                'strike_rate': 130.0
            },
            'bowling_stats': {
                'matches': 50,
                'wickets': 20,
                'average': 35.0,
                'economy': 8.0
            }
        }
        
        mock_process.return_value = mock_scrape.return_value
        mock_store.return_value = MagicMock()
        
        # Test
        result = self.manager.collect_player_data('http://example.com/player/1')
        self.assertIsNotNone(result)
        self.assertEqual(result['name'], 'Virat Kohli')
        
        # Verify calls
        mock_scrape.assert_called_once_with('http://example.com/player/1')
        mock_process.assert_called_once_with(mock_scrape.return_value)
        mock_store.assert_called_once_with(mock_process.return_value)
        
    @patch.object(ESPNCricinfoScraper, 'scrape_team_data')
    @patch.object(DataProcessor, 'process_team_data')
    @patch.object(StorageManager, 'store_team_data')
    def test_collect_team_data(self, mock_store, mock_process, mock_scrape):
        """Test collecting team data."""
        # Mock data
        mock_scrape.return_value = {
            'name': 'Mumbai Indians',
            'short_name': 'MI',
            'founded_year': 2008,
            'home_ground': 'Wankhede Stadium',
            'total_matches': 200,
            'wins': 120,
            'losses': 80,
            'win_percentage': 60.0
        }
        
        mock_process.return_value = mock_scrape.return_value
        mock_store.return_value = MagicMock()
        
        # Test
        result = self.manager.collect_team_data('http://example.com/team/1')
        self.assertIsNotNone(result)
        self.assertEqual(result['name'], 'Mumbai Indians')
        
        # Verify calls
        mock_scrape.assert_called_once_with('http://example.com/team/1')
        mock_process.assert_called_once_with(mock_scrape.return_value)
        mock_store.assert_called_once_with(mock_process.return_value)
        
    @patch.object(ESPNCricinfoScraper, 'get_season_match_urls')
    @patch.object(DataCollectionManager, 'collect_match_data')
    def test_collect_season_data(self, mock_collect_match, mock_get_urls):
        """Test collecting season data."""
        # Mock data
        mock_get_urls.return_value = [
            'http://example.com/match/1',
            'http://example.com/match/2',
            'http://example.com/match/3'
        ]
        
        mock_collect_match.side_effect = [
            {'match_number': '1'},
            {'match_number': '2'},
            {'match_number': '3'}
        ]
        
        # Test
        result = self.manager.collect_season_data('2023')
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['match_number'], '1')
        
        # Verify calls
        mock_get_urls.assert_called_once_with('2023')
        self.assertEqual(mock_collect_match.call_count, 3)
        
    @patch.object(StorageManager, 'get_recent_matches')
    @patch.object(ESPNCricinfoScraper, 'get_match_url')
    @patch.object(DataCollectionManager, 'collect_match_data')
    def test_update_recent_data(self, mock_collect_match, mock_get_url, mock_get_recent):
        """Test updating recent data."""
        # Mock data
        mock_get_recent.return_value = [
            MagicMock(match_number='1', season='2023'),
            MagicMock(match_number='2', season='2023')
        ]
        
        mock_get_url.side_effect = [
            'http://example.com/match/1',
            'http://example.com/match/2'
        ]
        
        mock_collect_match.side_effect = [
            {'match_number': '1'},
            {'match_number': '2'}
        ]
        
        # Test
        result = self.manager.update_recent_data(days=30)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['match_number'], '1')
        
        # Verify calls
        mock_get_recent.assert_called_once_with(30)
        self.assertEqual(mock_get_url.call_count, 2)
        self.assertEqual(mock_collect_match.call_count, 2)
        
    def test_cleanup(self):
        """Test cleanup method."""
        with patch.object(ESPNCricinfoScraper, 'cleanup') as mock_cleanup:
            self.manager.cleanup()
            mock_cleanup.assert_called_once() 