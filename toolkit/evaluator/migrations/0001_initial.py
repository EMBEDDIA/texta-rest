# Generated by Django 2.2.19 on 2021-03-24 17:54

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('elastic', '0008_indexsplitter_str_val'),
        ('core', '0013_auto_20210126_1041'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Evaluator',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('description', models.CharField(max_length=1000)),
                ('query', models.TextField(default='{"query": {"match_all": {}}}')),
                ('true_fact', models.CharField(max_length=1000)),
                ('predicted_fact', models.CharField(max_length=1000)),
                ('true_fact_value', models.CharField(default=None, max_length=1000, null=True)),
                ('predicted_fact_value', models.CharField(default=None, max_length=1000, null=True)),
                ('average_function', models.CharField(max_length=1000)),
                ('add_individual_results', models.BooleanField(default=True, null=True)),
                ('accuracy', models.FloatField(default=None, null=True)),
                ('precision', models.FloatField(default=None, null=True)),
                ('recall', models.FloatField(default=None, null=True)),
                ('f1_score', models.FloatField(default=None, null=True)),
                ('confusion_matrix', models.TextField(blank=True, default='[]', null=True)),
                ('n_true_classes', models.IntegerField(default=None, null=True)),
                ('n_predicted_classes', models.IntegerField(default=None, null=True)),
                ('n_total_classes', models.IntegerField(default=None, null=True)),
                ('document_count', models.IntegerField(default=None, null=True)),
                ('scroll_size', models.IntegerField(default=500, null=True)),
                ('es_timeout', models.IntegerField(default=10, null=True)),
                ('individual_results', models.TextField(default='{}')),
                ('memory_buffer', models.FloatField(default=2, null=True)),
                ('scores_imprecise', models.BooleanField(default=None, null=True)),
                ('evaluation_type', models.CharField(default=None, max_length=1000, null=True)),
                ('plot', models.FileField(null=True, upload_to='data/media', verbose_name='')),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('indices', models.ManyToManyField(default=None, to='elastic.Index')),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='core.Project')),
                ('task', models.OneToOneField(null=True, on_delete=django.db.models.deletion.SET_NULL, to='core.Task')),
            ],
        ),
    ]
