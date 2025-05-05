"""
Advanced settings for the IPL data scrapers.
"""

# Selenium WebDriver settings
SELENIUM_CONFIG = {
    "headless": True,  # Run browser in headless mode
    "page_load_timeout": 30,
    "implicit_wait": 10,
    "window_size": (1920, 1080),
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "driver_path": None,  # Will be set dynamically
    "download_driver": True,  # Whether to download the driver if not found
}

# Proxy settings
PROXY_CONFIG = {
    "enabled": True,
    "proxy_list": [
        # Add your proxy list here
        # Format: 'http://username:password@ip:port'
        # Example: 'http://user:pass@192.168.1.1:8080'
    ],
    "proxy_rotation_interval": 10,  # Number of requests before rotating proxy
    "max_fails": 3,  # Maximum number of failed requests before switching proxy
}

# Rate limiting settings
RATE_LIMIT_CONFIG = {
    "requests_per_minute": 30,
    "concurrent_requests": 3,
    "backoff_factor": 1.5,  # Multiplier for delay after failed requests
    "max_retries": 5,
    "retry_delay": 5,  # Base delay in seconds
}

# Robots.txt settings
ROBOTS_CONFIG = {
    "respect_robots_txt": True,
    "crawl_delay": 2,  # Default crawl delay in seconds
    "user_agent": "IPLDataScraper/1.0",  # Custom user agent for robots.txt
}

# Browser profiles for Selenium
BROWSER_PROFILES = {
    "chrome": {
        "arguments": [
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--start-maximized",
        ],
        "preferences": {
            "profile.default_content_setting_values.notifications": 2,
            "profile.managed_default_content_settings.images": 2,  # Disable images
        },
    }
}

# Cache settings
CACHE_CONFIG = {
    "enabled": True,
    "cache_dir": "cache",
    "cache_expiry": 3600,  # Cache expiry in seconds (1 hour)
}

# Logging settings
LOG_CONFIG = {
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "scraper.log",
    "max_log_size": 10485760,  # 10MB
    "backup_count": 5,
}
