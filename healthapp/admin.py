from django.contrib import admin

# Register your models here.
from healthapp.models import  HealthCustom




@admin.register(HealthCustom)
class HealthCustomAdmin(admin.ModelAdmin):
    pass