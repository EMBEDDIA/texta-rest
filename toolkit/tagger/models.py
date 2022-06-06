import io
import json
import logging
import os
import pathlib
import secrets
import tempfile
import zipfile
from typing import List, Union, Dict

from django.contrib.auth.models import User
from django.core import serializers
from django.db import models, transaction
from django.dispatch import receiver
from django.http import HttpResponse

from texta_embedding.embedding import W2VEmbedding
from texta_tagger.tagger import Tagger as TextTagger

from toolkit.constants import MAX_DESC_LEN
from toolkit.core.lexicon.models import Lexicon
from toolkit.core.project.models import Project
from toolkit.core.task.models import Task
from toolkit.elastic.choices import DEFAULT_SNOWBALL_LANGUAGE, get_snowball_choices
from toolkit.elastic.index.models import Index
from texta_elastic.searcher import EMPTY_QUERY
from toolkit.embedding.models import Embedding
from toolkit.settings import BASE_DIR, CELERY_LONG_TERM_TASK_QUEUE, INFO_LOGGER, RELATIVE_MODELS_PATH
from toolkit.helper_functions import load_stop_words
from toolkit.tools.lemmatizer import CeleryLemmatizer, ElasticAnalyzer
from toolkit.tagger import choices
from toolkit.elastic.tools.feedback import Feedback


class Tagger(models.Model):
    MODEL_TYPE = 'tagger'
    MODEL_JSON_NAME = "model.json"

    description = models.CharField(max_length=MAX_DESC_LEN)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    query = models.TextField(default=json.dumps(EMPTY_QUERY))
    fact_name = models.CharField(max_length=MAX_DESC_LEN, null=True)
    pos_label = models.CharField(max_length=MAX_DESC_LEN, null=True, default="", blank=True)
    indices = models.ManyToManyField(Index)
    fields = models.TextField(default=json.dumps([]))
    embedding = models.ForeignKey(Embedding, on_delete=models.SET_NULL, null=True, default=None)
    stop_words = models.TextField(default="[]")
    ignore_numbers = models.BooleanField(default=None, null=True)
    vectorizer = models.CharField(default=choices.DEFAULT_VECTORIZER, max_length=MAX_DESC_LEN)
    analyzer = models.CharField(default=choices.DEFAULT_ANALYZER, max_length=MAX_DESC_LEN)
    classifier = models.CharField(default=choices.DEFAULT_CLASSIFIER, max_length=MAX_DESC_LEN)
    negative_multiplier = models.FloatField(default=choices.DEFAULT_NEGATIVE_MULTIPLIER, blank=True)
    maximum_sample_size = models.IntegerField(default=choices.DEFAULT_MAX_SAMPLE_SIZE, blank=True)
    minimum_sample_size = models.IntegerField(default=choices.DEFAULT_MIN_SAMPLE_SIZE, blank=True)
    score_threshold = models.FloatField(default=choices.DEFAULT_SCORE_THRESHOLD, blank=True)
    snowball_language = models.CharField(choices=get_snowball_choices(), default=DEFAULT_SNOWBALL_LANGUAGE, null=True, max_length=MAX_DESC_LEN)
    detect_lang = models.BooleanField(default=False)
    precision = models.FloatField(default=None, null=True)
    recall = models.FloatField(default=None, null=True)
    f1_score = models.FloatField(default=None, null=True)
    num_features = models.IntegerField(default=None, null=True)
    num_examples = models.TextField(default="{}", null=True)
    confusion_matrix = models.TextField(default="[]", null=True, blank=True)
    scoring_function = models.CharField(default=choices.DEFAULT_SCORING_FUNCTION, max_length=MAX_DESC_LEN, null=True, blank=True)
    classes = models.TextField(default=json.dumps([]))

    model = models.FileField(null=True, verbose_name='', default=None)
    model_size = models.FloatField(default=None, null=True)
    plot = models.FileField(upload_to="data/media", null=True, verbose_name="")
    task = models.OneToOneField(Task, on_delete=models.SET_NULL, null=True)

    tagger_groups = models.TextField(default="[]", null=True, blank=True)

    balance = models.BooleanField(default=choices.DEFAULT_BALANCE)
    balance_to_max_limit = models.BooleanField(default=choices.DEFAULT_BALANCE_TO_MAX_LIMIT)


    def get_available_or_all_indices(self, indices: List[str] = None) -> List[str]:
        """
        Used in views where the user can select the indices they wish to use.
        Returns a list of index names from the ones that are in the project
        and in the indices parameter or all of the indices if it's None or empty.
        """
        if indices:
            indices = self.indices.filter(name__in=indices, is_open=True)
            if not indices:
                indices = self.project.indices.all()
        else:
            indices = self.indices.all()

        indices = [index.name for index in indices]
        indices = list(set(indices))  # Leave only unique names just in case.
        return indices


    def get_indices(self):
        return [index.name for index in self.indices.filter(is_open=True)]


    def set_confusion_matrix(self, x):
        self.confusion_matrix = json.dumps(x)


    def get_confusion_matrix(self):
        return json.loads(self.confusion_matrix)


    def __str__(self):
        return '{0} - {1}'.format(self.pk, self.description)


    def generate_name(self, name="tagger"):
        """
        Do not change this carelessly as import/export functionality depends on this.
        Returns full and relative filepaths for the intended models.
        Args:
            name: Name for the model to distinguish itself from others in the same directory.

        Returns: Full and relative file paths, full for saving the model object and relative for actual DB storage.
        """
        model_file_name = f'{name}_{str(self.pk)}_{secrets.token_hex(10)}'
        full_path = pathlib.Path(BASE_DIR) / RELATIVE_MODELS_PATH / "tagger" / model_file_name
        relative_path = pathlib.Path(RELATIVE_MODELS_PATH) / "tagger" / model_file_name
        return str(full_path), str(relative_path)


    def to_json(self) -> dict:
        serialized = serializers.serialize('json', [self])
        json_obj = json.loads(serialized)[0]["fields"]
        del json_obj["project"]
        del json_obj["author"]
        del json_obj["task"]
        return json_obj


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


    @staticmethod
    def import_resources(zip_file, request, pk) -> int:
        with transaction.atomic():
            with zipfile.ZipFile(zip_file, 'r') as archive:
                json_string = archive.read(Tagger.MODEL_JSON_NAME).decode()
                model_json = json.loads(json_string)
                indices = model_json.pop("indices")

                new_model = Tagger(**model_json)

                new_model.task = Task.objects.create(tagger=new_model, status=Task.STATUS_COMPLETED)
                new_model.author = User.objects.get(id=request.user.id)
                new_model.project = Project.objects.get(id=pk)
                new_model.save()  # Save the intermediate results.

                for index in indices:
                    index_model, is_created = Index.objects.get_or_create(name=index)
                    new_model.indices.add(index_model)

                full_tagger_path, relative_tagger_path = new_model.generate_name("tagger")
                with open(full_tagger_path, "wb") as fp:
                    path = pathlib.Path(model_json["model"]).name
                    fp.write(archive.read(path))
                    new_model.model.name = relative_tagger_path

                plot_name = pathlib.Path(model_json["plot"])
                path = plot_name.name
                new_model.plot.save(f'{secrets.token_hex(15)}.png', io.BytesIO(archive.read(path)))

                new_model.save()
                return new_model.id


    def get_resource_paths(self):
        """
        Return the full paths of every resource used by this model that lives
        on the filesystem.
        """
        return {"model": self.model.path, "plot": self.plot.path}


    def train(self):
        new_task = Task.objects.create(tagger=self, status='created')
        self.task = new_task
        self.save()
        from toolkit.tagger.tasks import start_tagger_task, train_tagger_task, save_tagger_results
        logging.getLogger(INFO_LOGGER).info(f"Celery: Starting task for training of tagger: {self.to_json()}")
        chain = start_tagger_task.s() | train_tagger_task.s() | save_tagger_results.s()
        transaction.on_commit(lambda: chain.apply_async(args=(self.pk,), queue=CELERY_LONG_TERM_TASK_QUEUE))


    def load_tagger(self, lemmatize: bool = False, use_logger: bool = True):
        """Loading tagger model from disc."""
        #if use_logger:
        #    logging.getLogger(INFO_LOGGER).info(f"Loading tagger with ID: {tagger_id} with params (lemmatize: {lemmatize})")
        # get lemmatizer/stemmer
        if self.snowball_language:
            lemmatizer = ElasticAnalyzer(language=self.snowball_language)
        elif lemmatize:
            lemmatizer = CeleryLemmatizer()
        else:
            lemmatizer = None
        # Load stop words
        stop_words = load_stop_words(self.stop_words)
        # load embedding
        if self.embedding:
            embedding = W2VEmbedding()
            embedding.load_django(self.embedding)
        else:
            embedding = False
        # load tagger
        tagger = TextTagger(embedding=embedding, mlp=lemmatizer, custom_stop_words=stop_words)
        tagger_loaded = tagger.load_django(self)
        # check if tagger gets loaded
        if not tagger_loaded:
            return None
        return tagger


    def apply_loaded_tagger(self, tagger: TextTagger, content: Union[str, Dict[str, str]], input_type: str = "text", feedback: bool = False):
        """Applying loaded tagger."""
        # check input type
        if input_type == 'doc':
            tagger_result = tagger.tag_doc(content)
        else:
            tagger_result = tagger.tag_text(content)
        # Result is false if binary tagger's prediction is false, but true otherwise
        # (for multiclass, the result is always true as one of the classes is always predicted)
        result = False if tagger_result["prediction"] == "false" else True
        # Use tagger description as tag for binary taggers and tagger prediction as tag for multiclass taggers
        tag = tagger.description if tagger_result["prediction"] in {"true", "false"} else tagger_result["prediction"]
        # create output dict
        prediction = {
            'tag': tag,
            'probability': tagger_result['probability'],
            'tagger_id': self.pk,
            'result': result
        }
        # add feedback if asked
        if feedback:
            logging.getLogger(INFO_LOGGER).info(f"Adding feedback for Tagger id: {self.pk}")
            project_pk = self.project.pk
            feedback_object = Feedback(project_pk, model_object=self)
            processed_text = tagger.text_processor.process(content)
            feedback_id = feedback_object.store(processed_text, prediction)
            feedback_url = f'/projects/{project_pk}/taggers/{self.pk}/feedback/'
            prediction['feedback'] = {'id': feedback_id, 'url': feedback_url}
        return prediction


@receiver(models.signals.post_delete, sender=Tagger)
def auto_delete_file_on_delete(sender, instance: Tagger, **kwargs):
    """
    Delete resources on the file-system upon tagger deletion.
    Triggered on individual-queryset Tagger deletion and the deletion
    of a TaggerGroup.
    """
    if instance.plot:
        if os.path.isfile(instance.plot.path):
            os.remove(instance.plot.path)

    if instance.model:
        if os.path.isfile(instance.model.path):
            os.remove(instance.model.path)


class TaggerGroup(models.Model):
    MODEL_JSON_NAME = "model.json"
    MODEL_TYPE = "tagger_group"

    description = models.CharField(max_length=MAX_DESC_LEN)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    fact_name = models.CharField(max_length=MAX_DESC_LEN)
    blacklisted_facts = models.TextField(default='[]')
    num_tags = models.IntegerField(default=0)
    minimum_sample_size = models.IntegerField(default=choices.DEFAULT_MIN_SAMPLE_SIZE)
    task = models.OneToOneField(Task, on_delete=models.SET_NULL, null=True)

    taggers = models.ManyToManyField(Tagger, default=None)


    def export_resources(self) -> HttpResponse:
        with tempfile.SpooledTemporaryFile(encoding="utf8") as tmp:
            with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as archive:
                # Write model object to zip as json
                model_json = self.to_json()
                model_json = json.dumps(model_json).encode("utf8")
                archive.writestr(self.MODEL_JSON_NAME, model_json)

                for item in self.get_resource_paths():
                    for file_path in item.values():
                        path = pathlib.Path(file_path)
                        archive.write(file_path, arcname=str(path.name))

            tmp.seek(0)
            return tmp.read()


    @staticmethod
    def import_resources(zip_file, request, pk) -> int:
        with transaction.atomic():
            with zipfile.ZipFile(zip_file, 'r') as archive:
                json_string = archive.read(Tagger.MODEL_JSON_NAME).decode()
                model_json: dict = json.loads(json_string)
                tg_data = {key: model_json[key] for key in model_json if key != 'taggers'}
                new_model = TaggerGroup(**tg_data)
                new_model.task = Task.objects.create(taggergroup=new_model, status=Task.STATUS_COMPLETED)
                new_model.author = User.objects.get(id=request.user.id)
                new_model.project = Project.objects.get(id=pk)
                new_model.save()  # Save the intermediate results.

                for tagger in model_json["taggers"]:
                    indices = tagger.pop("indices")
                    tagger_model = Tagger(**tagger)

                    tagger_model.task = Task.objects.create(tagger=tagger_model, status=Task.STATUS_COMPLETED)
                    tagger_model.author = User.objects.get(id=request.user.id)
                    tagger_model.project = Project.objects.get(id=pk)
                    tagger_model.save()

                    for index_name in indices:
                        index, is_created = Index.objects.get_or_create(name=index_name)
                        tagger_model.indices.add(index)

                    full_tagger_path, relative_tagger_path = tagger_model.generate_name("tagger")
                    with open(full_tagger_path, "wb") as fp:
                        path = pathlib.Path(tagger["model"]).name
                        fp.write(archive.read(path))
                        tagger_model.model.name = relative_tagger_path

                        plot_name = pathlib.Path(tagger["plot"])
                        path = plot_name.name
                        tagger_model.plot.save(f'{secrets.token_hex(15)}.png', io.BytesIO(archive.read(path)))
                        tagger_model.save()

                    new_model.taggers.add(tagger_model)

                new_model.save()
                return new_model.id


    def to_json(self) -> dict:
        serialized = serializers.serialize('json', [self])
        json_obj = json.loads(serialized)[0]["fields"]
        json_obj["taggers"] = [tg.to_json() for tg in self.taggers.all()]
        del json_obj["project"]
        del json_obj["author"]
        del json_obj["task"]
        return json_obj


    def __str__(self):
        return self.fact_name


    def get_resource_paths(self):
        container = []
        taggers = self.taggers.all()
        for tagger in taggers:
            container.append(tagger.get_resource_paths())
        return container


    def get_indices(self):
        # Retrieve project indices for checking if the Tagger Group index still exists
        project_indices = self.project.get_indices()
        # Retrieve the indices used for training the Tagger Group
        tg_indices = [index.name for index in self.taggers.first().indices.all()]
        # Get indices still present in the project
        tg_indices_in_project = list(set(tg_indices).intersection(set(project_indices)))
        # If no indices used for training are present in the current project, return all project indices
        if not tg_indices_in_project:
            logging.getLogger(INFO_LOGGER).info(f"[Tagger Group] Indices used for training ({tg_indices}) are not present in the current project. Using all the indices present in the project ({project_indices}).")
            return project_indices
        return tg_indices_in_project


@receiver(models.signals.pre_delete, sender=TaggerGroup)
def auto_delete_taggers_of_taggergroup(sender, instance: TaggerGroup, **kwargs):
    """
    Delete all the Taggers associated to the TaggerGroup before deletion
    to enforce a one-to-many behaviour. Triggered before the actual deletion.
    """
    instance.taggers.all().delete()
