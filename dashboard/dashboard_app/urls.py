from django.urls import path
from . import views

urlpatterns = [
    path('per-language/', views.per_language_sentiment, name='per_language'),
    path('per-language-data/', views.per_language_data, name='per_language_data'),
    path('live-feed/', views.live_feed, name='live_feed'),
    path('live-feed-data/', views.live_feed_data, name='live_feed_data'),
]
