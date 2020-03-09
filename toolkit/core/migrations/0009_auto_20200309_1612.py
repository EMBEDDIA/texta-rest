# Generated by Django 2.2.11 on 2020-03-09 14:12
import json

from django.db import migrations, models


def data_migration(apps, schema_editor):
    Project = apps.get_model('core', 'Project')
    for project in Project.objects.all():
        indices = list(project.indices)
        project.keeper = json.dumps(indices)
        project.save()


def final_solution(apps, schema_editor):
    Project = apps.get_model('core', 'Project')
    Index = apps.get_model('elastic', 'Index')

    for project in Project.objects.all():
        indices = json.loads(project.keeper)
        for index_name in indices:
            i, is_created = Index.objects.get_or_create(name=index_name)
            project.indices.add(i)
        project.save()


class Migration(migrations.Migration):
    dependencies = [
        ('elastic', '0003_index'),
        ('core', '0008_changes_to_task_api'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='keeper',
            field=models.TextField(default="[]"),
        ),

        migrations.RunPython(data_migration),

        migrations.RemoveField(
            model_name='project',
            name='indices',
        ),
        migrations.AddField(
            model_name='project',
            name='indices',
            field=models.ManyToManyField(default=None, to='elastic.Index'),
        ),

        migrations.RunPython(final_solution),

        migrations.RemoveField(
            model_name='project',
            name='keeper',
        ),
    ]
