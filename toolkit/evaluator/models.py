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
from toolkit.elastic.index.models import Index
from toolkit.elastic.tools.searcher import EMPTY_QUERY
from toolkit.settings import BASE_DIR, CELERY_LONG_TERM_TASK_QUEUE
from toolkit.evaluator import choices
#from toolkit.evaluator.tasks import evaluate_tags_task


class Evaluator(models.Model):
    MODEL_TYPE = 'evaluator'
    MODEL_JSON_NAME = "model.json"

    description = models.CharField(max_length=MAX_DESC_LEN)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    #fields = models.TextField(default=json.dumps([])) #are facts with different doc paths considered different?
    indices = models.ManyToManyField(Index, default=None)
    query = models.TextField(default=json.dumps(EMPTY_QUERY))

    true_fact = models.CharField(max_length=MAX_DESC_LEN, null=False)
    predicted_fact = models.CharField(max_length=MAX_DESC_LEN, null=False)
    true_fact_value = models.CharField(default=None, max_length=MAX_DESC_LEN, null=True)
    predicted_fact_value = models.CharField(default=None, max_length=MAX_DESC_LEN, null=True)

    average_function = models.CharField(null=False, max_length=MAX_DESC_LEN)

    accuracy = models.FloatField(default=None, null=True)
    precision = models.FloatField(default=None, null=True)
    recall = models.FloatField(default=None, null=True)
    f1_score = models.FloatField(default=None, null=True)
    confusion_matrix = models.TextField(default="[]", null=True, blank=True)


    #plot = models.FileField(upload_to='data/media', null=True, verbose_name='')

    task = models.OneToOneField(Task, on_delete=models.SET_NULL, null=True)


    def get_indices(self):
        return [index.name for index in self.indices.filter(is_open=True)]


    def to_json(self) -> dict:
        serialized = serializers.serialize('json', [self])
        #json_obj = json.loads(serialized)[0]["fields"]
        json_obj.pop("project")
        json_obj.pop("author")
        json_obj.pop("task")
        return json_obj


    @staticmethod
    def import_resources(zip_file, request, pk) -> int:
        with transaction.atomic():
            with zipfile.ZipFile(zip_file, 'r') as archive:
                json_string = archive.read(Evaluator.MODEL_JSON_NAME).decode()
                evaluator_json = json.loads(json_string)

                indices = evaluator_json.pop("indices")

                evaluator_model = Evaluator(**evaluator_json)

                evaluator_model.task = Task.objects.create(evaluator=evaluator_model, status=Task.STATUS_COMPLETED)
                evaluator_model.author = User.objects.get(id=request.user.id)
                evaluator_model.project = Project.objects.get(id=pk)

                evaluator_model.save()

                for index in indices:
                    index_model, is_created = Index.objects.get_or_create(name=index)
                    evaluator_model.indices.add(index_model)


                #plot_name = pathlib.Path(evaluator_json["plot"])
                #path = plot_name.name
                #evaluator_model.plot.save(f'{secrets.token_hex(15)}.png', io.BytesIO(archive.read(path)))

                evaluator_model.save()
                return evaluator_model.id


    def export_resources(self) -> HttpResponse:
        with tempfile.SpooledTemporaryFile(encoding="utf8") as tmp:
            with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as archive:
                # Write model object to zip as json
                model_json = self.to_json()
                model_json = json.dumps(model_json).encode("utf8")
                archive.writestr(self.MODEL_JSON_NAME, model_json)

                for file_path in self.get_resource_paths().values():
                    path = pathlib.Path(file_path)
                    archive.write(file_path, arcname=str(path.name))

            tmp.seek(0)
            return tmp.read()



    def __str__(self):
        return '{0} - {1}'.format(self.pk, self.description)

    """
    def evaluate_tags(self, indices, query, es_timeout = 10, bulk_size = 100):
        new_task = Task.objects.create(evaluator=self, status='created')
        self.task = new_task
        self.save()
        evaluate_tags_task.apply_async(args=(self.pk, indices, query, es_timeout, bulk_size), queue=CELERY_LONG_TERM_TASK_QUEUE)
     """
