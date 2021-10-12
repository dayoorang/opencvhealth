from django import forms
from healthapp.models import  HealthCustom




class HealthCustomForm(forms.ModelForm):
    class Meta:
        model = HealthCustom
        fields = ['set','exercise_1','exercise_2','exercise_3','repeats_1','repeats_2','repeats_3']

    def __init__(self, *args, **kwargs):
        super(HealthCustomForm, self).__init__(*args, **kwargs)

        for name, field in self.fields.items():
            field.widget.attrs.update({'class': 'form-control'})

        self.fields['set'].label = 'Sets'
        self.fields['exercise_1'].label = '첫번째 운동'
        self.fields['exercise_2'].label = '두번째 운동'
        self.fields['exercise_3'].label = '세번째 운동'

        # repeats label 제거
        self.fields['repeats_1'].label = ''
        self.fields['repeats_2'].label = ''
        self.fields['repeats_3'].label = ''

    def clean(self):
        cleaned_data = super().clean()
        exercise_1 = cleaned_data.get("exercise_1")
        exercise_2 = cleaned_data.get("exercise_2")
        exercise_3 = cleaned_data.get("exercise_3")
        list_1 = [exercise_1,exercise_2,exercise_3]

        # 전부 없음일경우 오류 발생

        # 없음을 제외한 다른 운동의 경우 중복될 경우 오류 발생
        if len(set(list_1)) != 3 and list_1.count('End') != 2 and list_1.count('End') != 3:
            self.add_error('exercise_1', "중복은 허용되지 않습니다")

        if 'End' in list_1:
            # 없음이 다른 운동 앞에 올 경우 문제 발생
            if list_1.index('End') == 0 and (list_1[list_1.index('End') + 1] != 'End' or list_1[list_1.index('End') + 2] != 'End'):
                self.add_error('exercise_1', "없음 다음에 운동이 올 수 없습니다. ")

            if list_1.index('End') == 1 and (list_1[list_1.index('End') + 1] != 'End'):
                self.add_error('exercise_1', "없음 다음에 운동이 올 수 없습니다. ")

            if list_1[0] == 'End' and list_1[1] == 'End' and list_1[2] == 'End':
                self.add_error('exercise_1', "적어도 하나의 이상 운동을 선택해 주십시오")
