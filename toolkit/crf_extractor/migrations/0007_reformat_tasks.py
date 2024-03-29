# Generated by Django 2.2.28 on 2022-08-08 11:56

from django.db import migrations, models


def transfer_crf_extractor_tasks(apps, schema_editor):
    # We can't import the Person model directly as it may be a newer
    # version than this migration expects. We use the historical version.

    CRFExtractor = apps.get_model('crf_extractor', 'CRFExtractor')
    for orm in CRFExtractor.objects.filter(task__isnull=False):
        task = getattr(orm, "task", None)
        if task:
            orm.tasks.add(orm.task)


class Migration(migrations.Migration):
    dependencies = [
        ('core', '0022_make_last_update_automatic'),
        ('crf_extractor', '0006_crfextractor_favorited_users'),
    ]

    operations = [
        migrations.AddField(
            model_name='crfextractor',
            name='tasks',
            field=models.ManyToManyField(to='core.Task'),
        ),

        migrations.RunPython(transfer_crf_extractor_tasks),

        migrations.RemoveField(
            model_name='crfextractor',
            name='task',
        ),

        migrations.AlterField(
            model_name='crfextractor',
            name='description',
            field=models.CharField(help_text='Description of the task to distinguish it from others.', max_length=1000),
        ),
    ]
