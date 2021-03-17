# Generated by Django 2.2.19 on 2021-03-17 10:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('evaluator', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='evaluator',
            name='n_predicted_classes',
            field=models.IntegerField(default=None, null=True),
        ),
        migrations.AddField(
            model_name='evaluator',
            name='n_total_classes',
            field=models.IntegerField(default=None, null=True),
        ),
        migrations.AddField(
            model_name='evaluator',
            name='n_true_classes',
            field=models.IntegerField(default=None, null=True),
        ),
    ]
