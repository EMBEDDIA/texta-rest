import io
import json
import os
import pathlib
import secrets
import tempfile
import zipfile

from django.contrib.auth.models import User
from django.core import serializers
from django.db import models, transaction
from django.dispatch import receiver
from django.http import HttpResponse

from toolkit.constants import MAX_DESC_LEN
from toolkit.core.project.models import Project
from toolkit.core.task.models import Task
from toolkit.elastic.searcher import EMPTY_QUERY
from toolkit.embedding.models import Embedding
from toolkit.helper_functions import apply_celery_task
from toolkit.settings import MODELS_DIR
from toolkit.torchtagger import choices


class TorchTagger(models.Model):
    MODEL_TYPE = 'torchtagger'
    MODEL_JSON_NAME = "model.json"

    description = models.CharField(max_length=MAX_DESC_LEN)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    fields = models.TextField(default=json.dumps([]))

    embedding = models.ForeignKey(Embedding, on_delete=models.CASCADE, default=None)

    query = models.TextField(default=json.dumps(EMPTY_QUERY))
    fact_name = models.CharField(max_length=MAX_DESC_LEN, null=True)
    minimum_sample_size = models.IntegerField(default=choices.DEFAULT_MIN_SAMPLE_SIZE)

    model_architecture = models.CharField(default=choices.MODEL_CHOICES[0][0], max_length=10)
    # seq_len = models.IntegerField(default=choices.DEFAULT_SEQ_LEN)
    # vocab_size = models.IntegerField(default=choices.DEFAULT_VOCAB_SIZE)
    num_epochs = models.IntegerField(default=choices.DEFAULT_NUM_EPOCHS)
    validation_ratio = models.FloatField(default=choices.DEFAULT_VALIDATION_SPLIT)
    negative_multiplier = models.FloatField(default=choices.DEFAULT_NEGATIVE_MULTIPLIER)
    maximum_sample_size = models.IntegerField(default=choices.DEFAULT_MAX_SAMPLE_SIZE)

    # RESULTS
    label_index = models.TextField(default=json.dumps({}))
    epoch_reports = models.TextField(default=json.dumps([]))
    accuracy = models.FloatField(default=None, null=True)
    training_loss = models.FloatField(default=None, null=True)
    precision = models.FloatField(default=None, null=True)
    recall = models.FloatField(default=None, null=True)
    f1_score = models.FloatField(default=None, null=True)
    model = models.FileField(null=True, verbose_name='', default=None)
    text_field = models.FileField(null=True, verbose_name='', default=None)
    plot = models.FileField(upload_to='data/media', null=True, verbose_name='')

    task = models.OneToOneField(Task, on_delete=models.SET_NULL, null=True)


    def get_indices(self):
        return [index.name for index in self.indices.all()]


    def to_json(self) -> dict:
        serialized = serializers.serialize('json', [self])
        json_obj = json.loads(serialized)[0]["fields"]
        json_obj.pop("embedding")
        json_obj.pop("project")
        json_obj.pop("author")
        json_obj.pop("task")
        return json_obj


    @staticmethod
    def import_resources(zip_file, request, pk) -> int:
        with transaction.atomic():
            with zipfile.ZipFile(zip_file, 'r') as archive:
                json_string = archive.read(TorchTagger.MODEL_JSON_NAME).decode()
                torch_and_embedding = json.loads(json_string)

                torchtagger_json = torch_and_embedding["torchtagger"]
                embedding_json = torch_and_embedding["embedding"]
                embedding_fields = embedding_json["fields"]
                extra_embeddings = embedding_json["embedding_extras"]
                torchtagger_json.pop("embedding", None)
                torchtagger_model = TorchTagger(**torchtagger_json)

                torchtagger_model.task = Task.objects.create(torchtagger=torchtagger_model, status=Task.STATUS_COMPLETED)
                torchtagger_model.author = User.objects.get(id=request.user.id)
                torchtagger_model.project = Project.objects.get(id=pk)

                embedding_model = Embedding.create_embedding_object(embedding_fields, request.user.id, pk)
                embedding_model = Embedding.add_file_to_embedding_object(archive, embedding_model, embedding_fields, "phraser", "phraser_model")
                embedding_model = Embedding.add_file_to_embedding_object(archive, embedding_model, embedding_fields, "embedding", "embedding_model")
                Embedding.save_embedding_extra_files(archive, embedding_model, embedding_fields, extra_paths=extra_embeddings)

                embedding_model.save()
                torchtagger_model.embedding = embedding_model
                torchtagger_model.save()

                new_tagger_name = torchtagger_model.generate_name("torchtagger")
                with open(new_tagger_name, "wb") as fp:
                    path = pathlib.Path(torchtagger_json["model"]).name
                    fp.write(archive.read(path))
                    torchtagger_model.model.name = new_tagger_name

                text_field = "{}_text_field".format(str(new_tagger_name))
                with open(text_field, "wb") as fp:
                    path = pathlib.Path(torchtagger_json["text_field"]).name
                    fp.write(archive.read(path))
                    torchtagger_model.text_field.name = text_field

                plot_name = pathlib.Path(torchtagger_json["plot"])
                path = plot_name.name
                torchtagger_model.plot.save(f'{secrets.token_hex(15)}.png', io.BytesIO(archive.read(path)))

                torchtagger_model.save()
                return torchtagger_model.id


    def export_resources(self) -> HttpResponse:
        with tempfile.SpooledTemporaryFile(encoding="utf8") as tmp:
            with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as archive:
                # Write model object to zip as json
                model_json = {"torchtagger": self.to_json(), "embedding": self.embedding.to_json()}
                model_json = json.dumps(model_json).encode("utf8")
                archive.writestr(self.MODEL_JSON_NAME, model_json)

                for file_path in self.get_resource_paths().values():
                    path = pathlib.Path(file_path)
                    archive.write(file_path, arcname=str(path.name))

                embedding = self.embedding.to_json()

                embedding_path = pathlib.Path(embedding["fields"]["embedding_model"])
                archive.write(str(embedding_path), arcname=str(embedding_path.name))

                phraser_path = pathlib.Path(embedding["fields"]["phraser_model"])
                archive.write(str(phraser_path), arcname=str(phraser_path.name))

                for file in embedding["embedding_extras"]:
                    archive.write(str(file), arcname=str(pathlib.Path(file).name))

            tmp.seek(0)
            return tmp.read()


    def generate_name(self, name):
        return os.path.join(MODELS_DIR, 'torchtagger', f'{name}_{str(self.pk)}_{secrets.token_hex(10)}')


    def __str__(self):
        return '{0} - {1}'.format(self.pk, self.description)


    def train(self):
        new_task = Task.objects.create(torchtagger=self, status='created')
        self.task = new_task
        self.save()
        from toolkit.torchtagger.tasks import train_torchtagger
        apply_celery_task(train_torchtagger, self.pk)


    def get_resource_paths(self):
        return {"plot": self.plot.path, "model": self.model.path, "text_field": self.text_field.path}


@receiver(models.signals.post_delete, sender=TorchTagger)
def auto_delete_torchtagger_on_delete(sender, instance: TorchTagger, **kwargs):
    """
    Delete resources on the file-system upon TorchTagger deletion.
    Triggered on individual model object and queryset TorchTagger deletion.
    """
    if instance.model:
        if os.path.isfile(instance.model.path):
            os.remove(instance.model.path)

    if instance.text_field:
        if os.path.isfile(instance.text_field.path):
            os.remove(instance.text_field.path)

    if instance.plot:
        if os.path.isfile(instance.plot.path):
            os.remove(instance.plot.path)
