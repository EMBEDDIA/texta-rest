# Generated by Django 2.1.15 on 2020-03-12 16:04

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('embedding', '0009_auto_20200309_1750'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='embeddingcluster',
            name='author',
        ),
        migrations.RemoveField(
            model_name='embeddingcluster',
            name='embedding',
        ),
        migrations.RemoveField(
            model_name='embeddingcluster',
            name='project',
        ),
        migrations.RemoveField(
            model_name='embeddingcluster',
            name='task',
        ),
        migrations.DeleteModel(
            name='EmbeddingCluster',
        ),
    ]
