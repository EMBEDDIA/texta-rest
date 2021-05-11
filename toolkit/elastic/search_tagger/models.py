import json
from django.db import models, transaction
from toolkit.constants import MAX_DESC_LEN
from django.contrib.auth.models import User
from toolkit.core.project.models import Project
from toolkit.elastic.tools.searcher import EMPTY_QUERY
from toolkit.elastic.index.models import Index
from toolkit.core.task.models import Task
from toolkit.settings import CELERY_LONG_TERM_TASK_QUEUE


class SearchQueryTagger(models.Model):
    description = models.CharField(max_length=MAX_DESC_LEN, default="")
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    indices = models.ManyToManyField(Index)
    query = models.TextField(default=json.dumps(EMPTY_QUERY))
    mapping_field = models.TextField(default=json.dumps([]))
    fact_name = models.TextField(default=json.dumps([]))
    fact_value = models.TextField(default=json.dumps([]))
    task = models.OneToOneField(Task, on_delete=models.SET_NULL, null=True)

    def get_indices(self):
        return [index.name for index in self.indices.filter(is_open=True)]

    def __str__(self):
        return self.description

    def process(self):
        from toolkit.elastic.search_tagger.tasks import apply_search_query_tagger_on_index

        new_task = Task.objects.create(search_tagger=self, status='created')
        self.task = new_task
        self.save()

        chain = apply_search_query_tagger_on_index.s()
        transaction.on_commit(lambda: chain.apply_async(args=(self.pk,), queue=CELERY_LONG_TERM_TASK_QUEUE))


class SearchFieldsTagger(models.Model):
    description = models.CharField(max_length=MAX_DESC_LEN, default="")
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    indices = models.ManyToManyField(Index)
    query = models.TextField(default=json.dumps(EMPTY_QUERY))
    fields = models.TextField(default=json.dumps([]))
    fact_name = models.TextField(default=json.dumps([]))
    task = models.OneToOneField(Task, on_delete=models.SET_NULL, null=True)

    def get_indices(self):
        return [index.name for index in self.indices.filter(is_open=True)]

    def __str__(self):
        return self.description

    def process(self):
        from toolkit.elastic.search_tagger.tasks import apply_search_fields_tagger_on_index

        new_task = Task.objects.create(search_tagger=self, status='created')
        self.task = new_task
        self.save()

        chain = apply_search_fields_tagger_on_index.s()
        transaction.on_commit(lambda: chain.apply_async(args=(self.pk,), queue=CELERY_LONG_TERM_TASK_QUEUE))
