"""
App configuration for data_collection.
"""

from django.apps import AppConfig


class DataCollectionConfig(AppConfig):
    """Configuration for the data_collection app."""
    
    default_auto_field = "django.db.models.BigAutoField"
    name = "data_collection"
    verbose_name = 'Data Collection'
    
    def ready(self):
        """Import signals when the app is ready."""
        import data_collection.signals  # noqa
