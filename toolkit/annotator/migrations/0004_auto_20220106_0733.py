# Generated by Django 2.2.25 on 2022-01-06 05:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('annotator', '0003_labelset_fact_names'),
    ]

    operations = [
        migrations.AlterField(
            model_name='labelset',
            name='fact_names',
            field=models.TextField(null=True),
        ),
    ]