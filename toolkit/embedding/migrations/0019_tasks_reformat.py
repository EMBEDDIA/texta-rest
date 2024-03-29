# Generated by Django 2.2.28 on 2022-08-08 12:28

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def transfer_embedding_tasks(apps, schema_editor):
    # We can't import the Person model directly as it may be a newer
    # version than this migration expects. We use the historical version.
    Embedding = apps.get_model('embedding', 'Embedding')
    for orm in Embedding.objects.filter(task__isnull=False):
        task = getattr(orm, "task", None)
        if task:
            orm.tasks.add(orm.task)


class Migration(migrations.Migration):
    dependencies = [
        ('core', '0022_make_last_update_automatic'),
        ('embedding', '0018_embedding_favorited_users'),
    ]

    operations = [

        migrations.AddField(
            model_name='embedding',
            name='tasks',
            field=models.ManyToManyField(to='core.Task'),
        ),

        migrations.RunPython(transfer_embedding_tasks),

        migrations.RemoveField(
            model_name='embedding',
            name='task',
        ),

        migrations.AlterField(
            model_name='embedding',
            name='author',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='embedding',
            name='description',
            field=models.CharField(help_text='Description of the task to distinguish it from others.', max_length=1000),
        ),
    ]
