# IPL Data Collection Module

This module provides functionality for collecting IPL data from ESPN Cricinfo and storing it in the database.

## Components

### 1. Scrapers
- `BaseScraper`: Abstract base class for web scrapers
- `ESPNCricinfoScraper`: Implementation for scraping data from ESPN Cricinfo

### 2. Processors
- `DataProcessor`: Handles data cleaning and normalization

### 3. Storage
- `StorageManager`: Manages database operations

### 4. Managers
- `DataCollectionManager`: Orchestrates the data collection pipeline

## Usage

### Django Management Command

```bash
# Collect data for a specific season
python manage.py collect_data --season 2023

# Collect data for a specific match
python manage.py collect_data --match http://example.com/match/1

# Update data for recent matches
python manage.py collect_data --update-recent --days 30
```

### Python API

```python
from data_collection.managers.data_collection_manager import DataCollectionManager

# Initialize the manager
manager = DataCollectionManager()

# Collect data for a season
matches = manager.collect_season_data('2023')

# Collect data for a match
match_data = manager.collect_match_data('http://example.com/match/1')

# Update recent data
recent_matches = manager.update_recent_data(days=30)

# Clean up resources
manager.cleanup()
```

## Testing

Run the tests using:

```bash
python manage.py test data_collection
```

## Dependencies

- Django
- Selenium
- fake-useragent
- pandas
- python-dateutil

## Configuration

The module uses the following settings in `settings.py`:

```python
# Selenium WebDriver settings
SELENIUM_DRIVER_PATH = 'path/to/chromedriver'
SELENIUM_HEADLESS = True

# ESPN Cricinfo settings
ESPNCRICINFO_BASE_URL = 'https://www.espncricinfo.com'
```

## Error Handling

The module includes comprehensive error handling and logging. Check the logs for details about any issues during data collection.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the tests
5. Submit a pull request 