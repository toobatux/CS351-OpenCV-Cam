from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('video_feed/', views.stream_video, name='video_feed'),
        # path('get_latest_alerts/', views.get_latest_alerts, name='get_latest_alerts'),
]