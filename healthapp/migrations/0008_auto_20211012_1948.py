# Generated by Django 3.2.8 on 2021-10-12 10:48

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('healthapp', '0007_auto_20211012_1932'),
    ]

    operations = [
        migrations.RenameField(
            model_name='healthcustom',
            old_name='excersie_1',
            new_name='exercise_1',
        ),
        migrations.RenameField(
            model_name='healthcustom',
            old_name='excersie_2',
            new_name='exercise_2',
        ),
        migrations.RenameField(
            model_name='healthcustom',
            old_name='excersie_3',
            new_name='exercise_3',
        ),
    ]
