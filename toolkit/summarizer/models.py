from django.db import models
from django.contrib.auth.models import User
from toolkit.core.project.models import Project
from toolkit.constants import MAX_DESC_LEN


class Summarizer(models.Model):
    text = models.CharField(max_length=MAX_DESC_LEN)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.text
