from django.contrib import admin

# Register your models here.
from healthapp.models import Health, HealthCustom


@admin.register(Health)
class HealthAdmin(admin.ModelAdmin):
    pass


@admin.register(HealthCustom)
class HealthCustomAdmin(admin.ModelAdmin):
    pass