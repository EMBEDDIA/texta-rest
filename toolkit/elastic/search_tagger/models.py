import json
from django.db import models, transaction
from toolkit.constants import MAX_DESC_LEN
from toolkit.core.project.models import Project
from toolkit.elastic.tools.searcher import EMPTY_QUERY
from toolkit.core.task.models import Task
from toolkit.settings import CELERY_LONG_TERM_TASK_QUEUE


class SearchTagger(models.Model):
    description = models.CharField(max_length=MAX_DESC_LEN, default="")
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    indices = models.TextField(default=json.dumps([]))
    query = models.TextField(default=json.dumps(EMPTY_QUERY))
    fields = models.TextField(default=json.dumps([]))
    task = models.OneToOneField(Task, on_delete=models.SET_NULL, null=True)

    def get_indices(self):
        return [index.name for index in self.indices.filter(is_open=True)]

    def __str__(self):
        return self.description

    def process(self):
        from toolkit.summarizer.tasks import apply_searchtagger_on_index, start_searchtagger_worker, end_searchtagger_task

        new_task = Task.objects.create(search_tagger=self, status='created')
        self.task = new_task
        self.save()

        chain = start_searchtagger_worker.s() | apply_searchtagger_on_index.s() | end_searchtagger_task.s()
        transaction.on_commit(lambda: chain.apply_async(args=(self.pk,), queue=CELERY_LONG_TERM_TASK_QUEUE))
