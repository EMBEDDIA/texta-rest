# Generated by Django 2.2.28 on 2022-08-08 13:17

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def transfer_rakun_tasks(apps, schema_editor):
    # We can't import the Person model directly as it may be a newer
    # version than this migration expects. We use the historical version.
    RakunExtractor = apps.get_model('rakun_keyword_extractor', 'RakunExtractor')
    for orm in RakunExtractor.objects.filter(task__isnull=False):
        task = getattr(orm, "task", None)
        if task:
            orm.tasks.add(orm.task)


class Migration(migrations.Migration):
    dependencies = [
        ('core', '0022_make_last_update_automatic'),
        ('rakun_keyword_extractor', '0002_rakunextractor_favorited_users'),
    ]

    operations = [

        migrations.AddField(
            model_name='rakunextractor',
            name='tasks',
            field=models.ManyToManyField(to='core.Task'),
        ),

        migrations.RunPython(transfer_rakun_tasks),

        migrations.RemoveField(
            model_name='rakunextractor',
            name='task',
        ),

        migrations.AlterField(
            model_name='rakunextractor',
            name='author',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='rakunextractor',
            name='description',
            field=models.CharField(help_text='Description of the task to distinguish it from others.', max_length=1000),
        ),
    ]
