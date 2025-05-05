# IPL Prediction System - Django Backend

This is the Django backend for the IPL Prediction System, providing RESTful API endpoints for match predictions, team and player data, and performance analysis.

## Features

- RESTful API endpoints for all data models
- Integration with ML models for predictions
- Swagger/OpenAPI documentation
- Admin interface for data management
- Caching for improved performance
- Authentication and authorization
- Comprehensive filtering and search capabilities

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

4. Create a superuser for admin access:
```bash
python manage.py createsuperuser
```

5. Run the development server:
```bash
python manage.py runserver
```

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/api/swagger/
- ReDoc: http://localhost:8000/api/redoc/

## API Endpoints

### Teams
- GET /api/teams/ - List all teams
- POST /api/teams/ - Create a new team
- GET /api/teams/{id}/ - Get team details
- PUT /api/teams/{id}/ - Update team
- DELETE /api/teams/{id}/ - Delete team

### Players
- GET /api/players/ - List all players
- POST /api/players/ - Create a new player
- GET /api/players/{id}/ - Get player details
- PUT /api/players/{id}/ - Update player
- DELETE /api/players/{id}/ - Delete player

### Matches
- GET /api/matches/ - List all matches
- POST /api/matches/ - Create a new match
- GET /api/matches/{id}/ - Get match details
- PUT /api/matches/{id}/ - Update match
- DELETE /api/matches/{id}/ - Delete match

### Predictions
- GET /api/predictions/ - List all predictions
- POST /api/predictions/ - Create a new prediction
- GET /api/predictions/{id}/ - Get prediction details
- POST /api/predictions/predict/ - Generate new predictions

### Player Performances
- GET /api/performances/ - List all performances
- POST /api/performances/ - Create a new performance
- GET /api/performances/{id}/ - Get performance details
- PUT /api/performances/{id}/ - Update performance
- DELETE /api/performances/{id}/ - Delete performance

## Filtering and Search

All list endpoints support filtering and search. For example:

- Filter teams by founded year: `/api/teams/?founded_year=2008`
- Search players by name: `/api/players/?search=virat`
- Filter matches by season: `/api/matches/?season=2023`
- Filter predictions by team: `/api/predictions/?team_id=1`

## Authentication

All API endpoints require authentication. Use one of the following methods:

1. Basic Authentication:
```bash
curl -u username:password http://localhost:8000/api/teams/
```

2. Session Authentication:
```bash
curl -X POST -d "username=your_username&password=your_password" http://localhost:8000/api-auth/login/
```

## Admin Interface

Access the admin interface at http://localhost:8000/admin/ using your superuser credentials.

## Development

To run tests:
```bash
python manage.py test
```

To check for code style issues:
```bash
flake8
```

## Deployment

For production deployment:

1. Set `DEBUG = False` in settings.py
2. Configure proper database settings
3. Set up static file serving
4. Configure proper security settings
5. Use a production-grade web server (e.g., Gunicorn)
6. Set up a reverse proxy (e.g., Nginx) 