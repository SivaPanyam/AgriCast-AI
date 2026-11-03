from django.urls import path
from .views import generate_forecast

urlpatterns = [
    path('forecast/', generate_forecast, name='generate_forecast'),
]