"""
Django management command for collecting IPL data.
"""

import logging
from django.core.management.base import BaseCommand
from data_collection.managers.data_collection_manager import DataCollectionManager

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Collect IPL data from ESPN Cricinfo'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--season',
            type=str,
            help='Collect data for a specific IPL season (e.g., 2023)'
        )
        parser.add_argument(
            '--match',
            type=str,
            help='Collect data for a specific match URL'
        )
        parser.add_argument(
            '--update-recent',
            action='store_true',
            help='Update data for recent matches'
        )
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Number of days to look back for recent matches (default: 30)'
        )
        
    def handle(self, *args, **options):
        """Handle the command."""
        try:
            manager = DataCollectionManager()
            
            if options['season']:
                self.stdout.write(f"Collecting data for season {options['season']}...")
                matches = manager.collect_season_data(options['season'])
                self.stdout.write(self.style.SUCCESS(
                    f"Successfully collected data for {len(matches)} matches"
                ))
                
            elif options['match']:
                self.stdout.write(f"Collecting data for match {options['match']}...")
                match_data = manager.collect_match_data(options['match'])
                if match_data:
                    self.stdout.write(self.style.SUCCESS(
                        f"Successfully collected data for match {match_data['match_number']}"
                    ))
                else:
                    self.stdout.write(self.style.ERROR("Failed to collect match data"))
                    
            elif options['update_recent']:
                self.stdout.write(f"Updating data for recent matches (last {options['days']} days)...")
                matches = manager.update_recent_data(options['days'])
                self.stdout.write(self.style.SUCCESS(
                    f"Successfully updated data for {len(matches)} matches"
                ))
                
            else:
                self.stdout.write(self.style.ERROR(
                    "Please specify one of: --season, --match, or --update-recent"
                ))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))
            logger.error(f"Error in collect_data command: {str(e)}")
            
        finally:
            manager.cleanup() 