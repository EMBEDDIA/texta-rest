# Generated by Django 2.2.28 on 2022-09-06 12:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('evaluator', '0004_evaluator_favorited_users'),
    ]

    operations = [
        migrations.AddField(
            model_name='evaluator',
            name='classes',
            field=models.TextField(default='[]'),
        ),
    ]
