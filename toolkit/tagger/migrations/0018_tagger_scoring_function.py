# Generated by Django 2.2.17 on 2021-01-29 13:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tagger', '0017_tagger_snowball_language'),
    ]

    operations = [
        migrations.AddField(
            model_name='tagger',
            name='scoring_function',
            field=models.CharField(blank=True, default='default', max_length=1000, null=True),
        ),
    ]
