# Generated by Django 2.2.19 on 2021-07-07 10:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tagger', '0026_tagger_minimum_sample_size'),
    ]

    operations = [
        migrations.AddField(
            model_name='tagger',
            name='pos_label',
            field=models.CharField(blank=True, default='', max_length=1000, null=True),
        ),
    ]
