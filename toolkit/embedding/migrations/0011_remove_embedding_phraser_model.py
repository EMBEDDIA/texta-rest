# Generated by Django 2.1.15 on 2020-03-25 07:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('embedding', '0010_auto_20200312_1804'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='embedding',
            name='phraser_model',
        ),
    ]
