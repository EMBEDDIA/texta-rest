import json
from django.db import models
from django.contrib.auth.models import User
from toolkit.core.project.models import Project
from toolkit.constants import MAX_DESC_LEN
from toolkit.elastic.index.models import Index
from toolkit.elastic.tools.searcher import EMPTY_QUERY


class Summarizer(models.Model):
    description = models.CharField(max_length=MAX_DESC_LEN)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    query = models.TextField(default=json.dumps(EMPTY_QUERY))
    indices = models.ManyToManyField(Index)
    fields = models.TextField(default=json.dumps([]))

    def __str__(self):
        return self.description
