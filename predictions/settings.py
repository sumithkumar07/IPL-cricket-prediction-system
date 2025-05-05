"""
Settings for Selenium and ESPN Cricinfo configuration.
"""

# Selenium Configuration
SELENIUM_CONFIG = {
    'driver_path': 'path/to/chromedriver.exe',  # Update this path
    'headless': True,  # Run browser in headless mode
    'timeout': 30,  # Page load timeout in seconds
    'wait_time': 10,  # Implicit wait time in seconds
}

# ESPN Cricinfo Configuration
ESPN_CRICINFO_CONFIG = {
    'base_url': 'https://www.espncricinfo.com',
    'series_url': '/series/indian-premier-league-{year}',
    'match_url': '/series/indian-premier-league-{year}-{match_id}',
    'player_url': '/player/{player_id}',
    'team_url': '/team/{team_id}',
}

# Data Collection Configuration
DATA_COLLECTION_CONFIG = {
    'start_year': 2008,  # IPL started in 2008
    'end_year': 2024,  # Current year
    'data_dir': 'data/raw',  # Directory to store raw data
    'processed_dir': 'data/processed',  # Directory to store processed data
    'backup_dir': 'data/backup',  # Directory for data backups
}

# Database Configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'name': 'ipl_prediction',
    'user': 'postgres',
    'password': 'your_password',  # Update this
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logs/data_collection.log',
            'formatter': 'standard',
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
    },
    'loggers': {
        '': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True
        },
    },
} 