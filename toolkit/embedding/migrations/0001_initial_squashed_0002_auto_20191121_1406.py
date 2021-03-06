# Generated by Django 2.1.14 on 2019-12-06 07:00

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    replaces = [('embedding', '0001_initial'), ('embedding', '0002_auto_20191121_1406')]

    dependencies = [
        ('core', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Embedding',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('description', models.CharField(max_length=100)),
                ('query', models.TextField(default='{"query": {"match_all": {}}}')),
                ('fields', models.TextField(default='[]')),
                ('num_dimensions', models.IntegerField(default=100)),
                ('min_freq', models.IntegerField(default=10)),
                ('vocab_size', models.IntegerField(default=0)),
                ('location', models.TextField(default='{}')),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='core.Project')),
                ('task', models.OneToOneField(null=True, on_delete=django.db.models.deletion.SET_NULL, to='core.Task')),
            ],
        ),
        migrations.CreateModel(
            name='EmbeddingCluster',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('description', models.CharField(max_length=1000)),
                ('num_clusters', models.IntegerField(default=100)),
                ('location', models.TextField(default='{}')),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('embedding', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='embedding.Embedding')),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='core.Project')),
                ('task', models.OneToOneField(null=True, on_delete=django.db.models.deletion.SET_NULL, to='core.Task')),
            ],
        ),
        migrations.AlterField(
            model_name='embedding',
            name='description',
            field=models.CharField(max_length=1000),
        ),
    ]
