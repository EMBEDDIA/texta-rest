# Generated by Django 2.1.15 on 2020-04-29 07:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('topic_analyzer', '0002_clusteringresult_embedding'),
    ]

    operations = [
        migrations.AddField(
            model_name='clusteringresult',
            name='significant_words_filter',
            field=models.CharField(default='[0-9]+', max_length=100),
        ),
    ]
