from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models

# Create your models here.
class Health(models.Model):
    repeats = models.PositiveIntegerField(blank=False, null=False, default=1) # 운동 반복 횟수
    set = models.PositiveIntegerField(blank=False, null=False, default=1) # 운동 세트 횟수
    created_at = models.DateTimeField(auto_now_add=True)



EXERCISE_CHOICES = (
            ('Curl', '아령들기' ),
            ('Squat', '스쿼트' ),
            ('Push Up', '팔굽혀펴기'),
            ('없음', '없음'),
)


class HealthCustom(models.Model):
    set = models.PositiveIntegerField(blank=False, null=False, default=1) # 운동 세트 횟수
    created_at = models.DateTimeField(auto_now_add=True)
    exercise_1 = models.CharField(max_length=100, blank=False, null=False, choices = EXERCISE_CHOICES)
    exercise_2 = models.CharField(max_length=100, blank=False, null=False, choices = EXERCISE_CHOICES)
    exercise_3 = models.CharField(max_length=100, blank=False, null=False, choices = EXERCISE_CHOICES)
    repeats_1 = models.PositiveIntegerField(blank=False, null=False, default=1) # 운동 반복 횟수
    repeats_2 = models.PositiveIntegerField(blank=False, null=False, default=1) # 운동 반복 횟수
    repeats_3 = models.PositiveIntegerField(blank=False, null=False, default=1) # 운동 반복 횟수

