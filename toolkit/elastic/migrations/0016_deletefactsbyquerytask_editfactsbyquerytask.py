# Generated by Django 2.2.24 on 2021-09-13 07:45

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0018_delete_phrase'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('elastic', '0015_breakup_characters'),
    ]

    operations = [
        migrations.CreateModel(
            name='EditFactsByQueryTask',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('description', models.CharField(default='', max_length=1000)),
                ('scroll_size', models.IntegerField(default=500, help_text='How many documents should be sent into Elasticsearch in a single batch for update.')),
                ('es_timeout', models.IntegerField(default=15, help_text='How many seconds should be allowed for the the update request to Elasticsearch.')),
                ('query', models.TextField(default='{"query": {"match_all": {}}}')),
                ('target_facts', models.TextField(help_text='Which facts to select for editing.')),
                ('fact', models.TextField(help_text='End result of the selected facts.')),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('indices', models.ManyToManyField(to='elastic.Index')),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='core.Project')),
                ('task', models.OneToOneField(null=True, on_delete=django.db.models.deletion.SET_NULL, to='core.Task')),
            ],
        ),
        migrations.CreateModel(
            name='DeleteFactsByQueryTask',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('description', models.CharField(default='', max_length=1000)),
                ('scroll_size', models.IntegerField(default=500, help_text='How many documents should be sent into Elasticsearch in a single batch for update.')),
                ('es_timeout', models.IntegerField(default=15, help_text='How many seconds should be allowed for the the update request to Elasticsearch.')),
                ('query', models.TextField(default='{"query": {"match_all": {}}}')),
                ('facts', models.TextField()),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('indices', models.ManyToManyField(to='elastic.Index')),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='core.Project')),
                ('task', models.OneToOneField(null=True, on_delete=django.db.models.deletion.SET_NULL, to='core.Task')),
            ],
        ),
    ]
