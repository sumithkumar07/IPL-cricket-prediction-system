# IPL Cricket Prediction System

A comprehensive system for predicting IPL cricket match outcomes using machine learning and data analysis.

## Project Overview

This project combines machine learning models, data analysis, and a Django backend to provide accurate predictions for IPL cricket matches. The system includes:

- Data collection and processing
- Machine learning models for match predictions
- Django REST API backend
- Integration with Large Language Models for prediction explanations

## Features

### Data Collection and Processing
- Web scraping from multiple sources (Cricbuzz, ESPN Cricinfo, IPL T20)
- Data cleaning and preprocessing
- Historical data analysis
- Feature engineering

### Machine Learning Models
- Ensemble model combining multiple algorithms
- Time series analysis for performance trends
- Player performance prediction
- Team strength analysis
- Weather and pitch condition impact analysis

### Backend API
- RESTful API endpoints
- Authentication and authorization
- Swagger/OpenAPI documentation
- Admin interface for data management
- Caching for improved performance

### Prediction System
- Match winner prediction
- Score prediction
- Player performance prediction
- Weather impact analysis
- Pitch condition analysis
- LLM-powered prediction explanations

## Project Structure

```
ipl/
├── data/                    # Raw and processed data
├── ml_model/               # Machine learning models
│   ├── ensemble.py         # Ensemble model implementation
│   ├── features.py         # Feature engineering
│   ├── llm_reasoning.py    # LLM integration
│   └── time_series.py      # Time series analysis
├── ipl_backend/            # Django backend
│   ├── settings.py         # Django settings
│   ├── urls.py            # Main URL configuration
│   └── wsgi.py            # WSGI configuration
├── predictions/            # Django app
│   ├── models.py          # Database models
│   ├── views.py           # API views
│   ├── serializers.py     # API serializers
│   └── urls.py            # App URL configuration
├── scripts/                # Utility scripts
│   ├── data_collector.py  # Data collection
│   ├── process_data.py    # Data processing
│   └── train_models.py    # Model training
└── tests/                 # Test files
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/sumithkumar07/IPL-cricket-prediction-system.git
cd IPL-cricket-prediction-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the database:
```bash
python manage.py makemigrations
python manage.py migrate
```

5. Create a superuser:
```bash
python manage.py createsuperuser
```

6. Run the development server:
```bash
python manage.py runserver
```

## API Documentation

Once the server is running, access the API documentation at:
- Swagger UI: http://localhost:8000/api/swagger/
- ReDoc: http://localhost:8000/api/redoc/

### Available Endpoints

- `/api/teams/` - Team information
- `/api/players/` - Player information
- `/api/matches/` - Match data
- `/api/predictions/` - Match predictions
- `/api/performances/` - Player performances

## Model Training

To train the models:

1. Collect data:
```bash
python scripts/data_collector.py
```

2. Process the data:
```bash
python scripts/process_data.py
```

3. Train the models:
```bash
python scripts/train_models.py
```

## Usage

1. Access the admin interface at http://localhost:8000/admin/
2. Use the API endpoints for predictions
3. View prediction results and explanations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data sources: Cricbuzz, ESPN Cricinfo, IPL T20
- Machine learning libraries: scikit-learn, pandas, numpy
- Web framework: Django
- API documentation: drf-yasg

## Contact

For any queries or suggestions, please contact:
- Email: sumithluckey@gmail.com
- GitHub: [sumithkumar07](https://github.com/sumithkumar07) 