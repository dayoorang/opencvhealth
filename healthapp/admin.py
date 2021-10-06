from django.contrib import admin

# Register your models here.
from healthapp.models import Health


@admin.register(Health)
class HealthAdmin(admin.ModelAdmin):
    pass