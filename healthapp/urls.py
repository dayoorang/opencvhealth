from django.contrib import admin
from django.urls import path, include

from healthapp.views import detectme, HealthCreationView, TrainingView, HealthDeleteView, HealthCompleteView

app_name = 'healthmapp'

urlpatterns = [
    path('training/<int:pk>',TrainingView.as_view(), name='training'),
    path('detectme/<int:pk>', detectme, name='detectme'),
    path('delete/<int:pk>', HealthDeleteView.as_view(), name='delete'),
    path('complete/<int:pk>', HealthCompleteView.as_view(), name='complete'),

    path('', HealthCreationView.as_view(), name='health'),

]
