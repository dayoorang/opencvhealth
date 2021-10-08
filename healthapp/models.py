from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models

# Create your models here.
class Health(models.Model):
    repeats = models.PositiveIntegerField(blank=False, null=False, default=1) # 운동 반복 횟수
    set = models.PositiveIntegerField(blank=False, null=False, default=1) # 운동 세트 횟수
    created_at = models.DateTimeField(auto_now_add=True)
