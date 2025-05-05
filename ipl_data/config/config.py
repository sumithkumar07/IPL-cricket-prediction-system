"""
Configuration settings for IPL data scraping.
"""

# Base URLs
IPLT20_BASE_URL = "https://www.iplt20.com"
ESPNCRICINFO_BASE_URL = "https://www.espncricinfo.com"
CRICBUZZ_BASE_URL = "https://www.cricbuzz.com"

# IPL 2024 specific URLs
IPL_2024_ID = "1345038"  # ESPNCricinfo series ID for IPL 2024
IPL_SEASON = "2024"

# IPLt20.com endpoints
IPLT20_ENDPOINTS = {
    "matches": f"{IPLT20_BASE_URL}/matches/schedule/men",
    "stats": f"{IPLT20_BASE_URL}/stats/{IPL_SEASON}",
    "points_table": f"{IPLT20_BASE_URL}/points-table/men",
    "teams": f"{IPLT20_BASE_URL}/teams",
    "news": f"{IPLT20_BASE_URL}/news",
}

# ESPNCricinfo endpoints
ESPNCRICINFO_ENDPOINTS = {
    "series": f"{ESPNCRICINFO_BASE_URL}/series/indian-premier-league-{IPL_SEASON}-{IPL_2024_ID}",
    "matches": f"{ESPNCRICINFO_BASE_URL}/series/indian-premier-league-{IPL_SEASON}-{IPL_2024_ID}/match-schedule",
    "stats": f"{ESPNCRICINFO_BASE_URL}/series/indian-premier-league-{IPL_SEASON}-{IPL_2024_ID}/stats",
    "points_table": f"{ESPNCRICINFO_BASE_URL}/series/indian-premier-league-{IPL_SEASON}-{IPL_2024_ID}/points-table",
    "teams": f"{ESPNCRICINFO_BASE_URL}/series/indian-premier-league-{IPL_SEASON}-{IPL_2024_ID}/squads",
}

# Cricbuzz endpoints
CRICBUZZ_ENDPOINTS = {
    "series": f"{CRICBUZZ_BASE_URL}/cricket-series/5945/indian-premier-league-{IPL_SEASON}",
    "matches": f"{CRICBUZZ_BASE_URL}/cricket-series/5945/indian-premier-league-{IPL_SEASON}/matches",
    "stats": f"{CRICBUZZ_BASE_URL}/cricket-series/5945/indian-premier-league-{IPL_SEASON}/stats",
    "points_table": f"{CRICBUZZ_BASE_URL}/cricket-series/5945/indian-premier-league-{IPL_SEASON}/points-table",
    "teams": f"{CRICBUZZ_BASE_URL}/cricket-series/5945/indian-premier-league-{IPL_SEASON}/teams",
}

# Updated selectors for IPLt20.com
IPLT20_SELECTORS = {
    "matches": {
        "container": "div.fixture",
        "match_card": "div.fixture__card",
        "teams": "span.fixture__team-name",
        "date": "span.fixture__date",
        "time": "span.fixture__time",
        "venue": "span.fixture__venue",
        "status": "span.fixture__status",
    },
    "stats": {
        "batting_table": "table.standings-table",
        "bowling_table": "table.standings-table--bowling",
        "player_name": "td.standings-table__player",
        "team_name": "td.standings-table__team",
    },
    "points_table": {
        "table": "table.standings-table",
        "team_name": "td.standings-table__team",
        "matches": "td.standings-table__matches",
        "points": "td.standings-table__points",
        "nrr": "td.standings-table__nrr",
    },
    "teams": {
        "container": "div.teams-list",
        "team_card": "div.teams-list__team",
        "team_name": "h2.teams-list__team-name",
        "captain": "div.teams-list__captain",
        "coach": "div.teams-list__coach",
    },
}

# Updated selectors for ESPNCricinfo
ESPNCRICINFO_SELECTORS = {
    "matches": {
        "container": "div.ds-grow",
        "match_card": "div.ds-border-b.ds-border-line",
        "teams": "p.ds-text-tight-m",
        "date": "span.ds-text-tight-xs",
        "time": "div.ds-text-tight-s",
        "venue": "div.ds-text-tight-xs",
        "status": "span.ds-text-tight-s",
    },
    "stats": {
        "batting_table": "table.ds-w-full.ds-table.ds-table-md.ds-table-auto.ds-w-full.ds-overflow-scroll",
        "bowling_table": "table.ds-w-full.ds-table.ds-table-md.ds-table-auto",
        "player_name": "td.ds-min-w-max",
        "team_name": "td.ds-min-w-max span",
    },
    "points_table": {
        "table": "table.ds-w-full.ds-table.ds-table-xs.ds-table-auto",
        "team_name": "span.ds-text-tight-s",
        "matches": "td.ds-w-0 div",
        "points": "td.ds-text-right",
        "nrr": "td.ds-text-right",
    },
    "teams": {
        "container": "div.ds-mt-3",
        "team_card": "div.ds-border.ds-border-line",
        "team_name": "h2.ds-text-title-s",
        "players": "div.ds-text-tight-m",
    },
}

# Updated selectors for Cricbuzz
CRICBUZZ_SELECTORS = {
    "matches": {
        "container": "div.cb-col.cb-col-100.cb-series-matches",
        "match_card": "div.cb-mtch-lst.cb-col.cb-col-100.cb-tms-itm",
        "teams": "div.cb-col-60.cb-col.cb-mtch-info",
        "date": "div.cb-col-40.cb-col",
        "venue": "div.text-gray",
    },
    "stats": {
        "batting_table": "table.table.cb-series-stats",
        "bowling_table": "table.table.cb-series-stats",
        "player_name": "td.text-left",
        "team_name": "td.text-left",
    },
    "points_table": {
        "table": "table.table.cb-series-points",
        "team_name": "td.text-left",
        "matches": "td.text-right",
        "points": "td.text-right",
        "nrr": "td.text-right",
    },
    "teams": {
        "container": "div.cb-col-100.cb-col",
        "team_card": "div.cb-team-card.cb-col-33.cb-col",
        "team_name": "h2.cb-team-name",
        "players": "div.cb-team-players",
    },
}

# Enhanced request headers with rotating User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

# Enhanced request headers
REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

# Cache settings
CACHE_EXPIRY = 1800  # 30 minutes in seconds
CACHE_DIR = "cache"

# Rate limiting settings
REQUEST_DELAY = 3  # seconds between requests
MAX_RETRIES = 5  # maximum number of retries for failed requests
RETRY_DELAY = 10  # seconds between retries
RETRY_STATUS_CODES = [403, 429, 500, 502, 503, 504]  # Status codes to retry on

# Output settings
DATA_DIR = "ipl_data/data/raw"
LOG_DIR = "logs"

# Logging Configuration
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/scraper.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "standard",
        },
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "": {"handlers": ["console", "file"], "level": "INFO", "propagate": True},
    },
}

# Selenium WebDriver Settings
WEBDRIVER_SETTINGS = {
    "headless": True,
    "window_size": (1920, 1080),
    "page_load_timeout": 30,
    "implicit_wait": 10,
    "arguments": [
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-notifications",
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        "--start-maximized",
    ],
}
