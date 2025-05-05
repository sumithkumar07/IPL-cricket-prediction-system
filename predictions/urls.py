"""
URL configuration for IPL prediction system API.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from . import views

# Create a router and register our viewsets with it
router = DefaultRouter()
router.register(r'teams', views.TeamViewSet)
router.register(r'players', views.PlayerViewSet)
router.register(r'matches', views.MatchViewSet)
router.register(r'performances', views.PlayerPerformanceViewSet)
router.register(r'predictions', views.PredictionViewSet)
router.register(r'player-predictions', views.PlayerPredictionViewSet)

# Schema view for Swagger documentation
schema_view = get_schema_view(
    openapi.Info(
        title="IPL Prediction System API",
        default_version='v1',
        description="API for IPL match predictions and analysis",
        terms_of_service="https://www.example.com/terms/",
        contact=openapi.Contact(email="contact@example.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

# The API URLs are now determined automatically by the router
urlpatterns = [
    # API endpoints
    path('', include(router.urls)),
    
    # Swagger documentation
    path('swagger<format>/', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
] 