# Generated by Django 2.2.28 on 2022-12-07 12:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0023_userprofile_uuid'),
        ('tagger', '0032_reformat_tasks'),
    ]

    operations = [
        migrations.AddField(
            model_name='taggergroup',
            name='ner_lexicons',
            field=models.ManyToManyField(default=[], to='core.Lexicon'),
        ),
        migrations.AddField(
            model_name='taggergroup',
            name='use_taggers_as_ner_filter',
            field=models.BooleanField(default=True),
        ),
    ]
