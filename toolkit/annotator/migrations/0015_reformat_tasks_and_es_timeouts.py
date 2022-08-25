# Generated by Django 2.2.28 on 2022-08-08 11:12

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def transfer_annotator_tasks(apps, schema_editor):
    # We can't import the Person model directly as it may be a newer
    # version than this migration expects. We use the historical version.
    Annotator = apps.get_model('annotator', 'Annotator')
    for orm in Annotator.objects.filter(task__isnull=False):
        task = getattr(orm, "task", None)
        if task:
            orm.tasks.add(orm.task)


class Migration(migrations.Migration):
    dependencies = [
        ('core', '0022_make_last_update_automatic'),
        ('annotator', '0014_annotatorgroup_project'),
    ]

    operations = [
        migrations.AddField(
            model_name='annotator',
            name='tasks',
            field=models.ManyToManyField(to='core.Task'),
        ),

        migrations.RunPython(transfer_annotator_tasks),

        migrations.RemoveField(
            model_name='annotator',
            name='task',
        ),

        migrations.AlterField(
            model_name='annotator',
            name='bulk_size',
            field=models.IntegerField(default=100, help_text='How many documents should be sent into Elasticsearch in a single batch for update.', validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(500)]),
        ),
        migrations.AlterField(
            model_name='annotator',
            name='es_timeout',
            field=models.IntegerField(default=10, help_text='How many seconds should be allowed for the the update request to Elasticsearch.', validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(100)]),
        ),

        migrations.AlterField(
            model_name='annotator',
            name='author',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL),
        ),

        migrations.AlterField(
            model_name='annotator',
            name='description',
            field=models.CharField(help_text='Description of the task to distinguish it from others.', max_length=1000),
        ),
    ]
