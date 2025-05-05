from setuptools import setup, find_packages

setup(
    name="ipl_data",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "beautifulsoup4",
        "pandas",
        "selenium>=4.9.0",
        "lxml",
        "fake-useragent",
        "python-dotenv",
        "webdriver-manager",
        "undetected-chromedriver>=3.5.0",
    ],
    python_requires=">=3.7",
)
