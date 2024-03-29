# Generated by Django 3.1 on 2022-04-12 14:07

from django.db import migrations
import multiselectfield.db.fields


class Migration(migrations.Migration):

    dependencies = [
        ('crf_extractor', '0003_auto_20211013_1158'),
    ]

    operations = [
        migrations.AlterField(
            model_name='crfextractor',
            name='context_feature_fields',
            field=multiselectfield.db.fields.MultiSelectField(choices=[('text', 'text'), ('lemmas', 'lemmas'), ('pos_tags', 'pos_tags'), ('word_features', 'word_features')], max_length=34),
        ),
        migrations.AlterField(
            model_name='crfextractor',
            name='feature_fields',
            field=multiselectfield.db.fields.MultiSelectField(choices=[('text', 'text'), ('lemmas', 'lemmas'), ('pos_tags', 'pos_tags'), ('word_features', 'word_features')], max_length=34),
        ),
    ]
