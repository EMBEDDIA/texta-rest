# Generated by Django 2.1.14 on 2019-11-26 12:52

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('tagger', '0008_model_path_migration'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='tagger',
            name='location',
        )
    ]
