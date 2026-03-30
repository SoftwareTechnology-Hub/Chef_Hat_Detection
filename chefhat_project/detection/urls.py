from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="detection_index"),
    path("alerts/", views.alerts, name="detection_alerts"),
    path("api/detect/", views.detect, name="detection_api"),
]
