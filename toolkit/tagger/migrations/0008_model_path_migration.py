# Generated by Django 2.1.14 on 2019-11-26 12:52
import json

from django.db import migrations


def transfer_existing_tagger_model_path(apps, schema_editor):
    """
    Iterating through objects.all() is pretty expensive usually
    as it makes multiple queries to the DB but in this case it cant
    be helped.
    """
    tagger_models = apps.get_model("tagger", "Tagger")
    for tagger in tagger_models.objects.all():
        if tagger.location:
            model_path = json.loads(tagger.location)["tagger"]
            tagger.model.name = model_path
            tagger.save()
        else:
            tagger.delete()


class Migration(migrations.Migration):
    dependencies = [
        ('tagger', '0007_add_model_path_field'),
    ]

    operations = [
        migrations.RunPython(transfer_existing_tagger_model_path)
    ]
