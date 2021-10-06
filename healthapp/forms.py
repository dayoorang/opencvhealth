from django import forms
from healthapp.models import Health


class HealthForm(forms.ModelForm):
    class Meta:
        model = Health
        fields = ['repeats','set']