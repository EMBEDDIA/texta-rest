# Generated by Django 2.2.27 on 2022-03-24 15:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tagger', '0028_tagger_analyzer'),
    ]

    operations = [
        migrations.AddField(
            model_name='taggergroup',
            name='blacklisted_facts',
            field=models.TextField(default='[]'),
        ),
    ]