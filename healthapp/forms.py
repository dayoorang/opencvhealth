from django import forms
from healthapp.models import Health


class HealthForm(forms.ModelForm):
    class Meta:
        model = Health
        fields = ['repeats','set']

    def __init__(self, *args, **kwargs):
        super(HealthForm, self).__init__(*args, **kwargs)

        for name, field in self.fields.items():
            field.widget.attrs.update({'class': 'form-control'})

        self.fields['repeats'].label = 'Reps'
        self.fields['set'].label = 'Sets'