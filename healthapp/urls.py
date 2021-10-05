from django.contrib import admin
from django.urls import path, include

from healthapp.views import detectme, home

app_name = 'healthmapp'

urlpatterns = [
    path('home/',home, name='home'),
    path('detectme/', detectme, name='detectme'),

]
