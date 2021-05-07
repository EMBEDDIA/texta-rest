import json

from django.urls import reverse
from rest_framework import serializers

from .models import ApplyStemmerWorker
from ..choices import DEFAULT_SNOWBALL_LANGUAGE, get_snowball_choices
from ..index.serializers import IndexSerializer
from ...core.task.serializers import TaskSerializer


class SnowballSerializer(serializers.Serializer):
    text = serializers.CharField()
    language = serializers.ChoiceField(choices=get_snowball_choices(), default=DEFAULT_SNOWBALL_LANGUAGE)


class ApplySnowballSerializer(serializers.ModelSerializer):
    description = serializers.CharField()
    indices = IndexSerializer(many=True, default=[])
    author_username = serializers.CharField(source='author.username', read_only=True, required=False)
    task = TaskSerializer(read_only=True, required=False)
    url = serializers.SerializerMethodField()
    query = serializers.JSONField(help_text='Query in JSON format', required=False)


    def get_url(self, obj):
        default_version = "v2"
        index = reverse(f"{default_version}:apply_snowball-detail", kwargs={"project_pk": obj.project.pk, "pk": obj.pk})
        if "request" in self.context:
            request = self.context["request"]
            url = request.build_absolute_uri(index)
            return url
        else:
            return None


    def to_representation(self, instance: ApplyStemmerWorker):
        data = super(ApplySnowballSerializer, self).to_representation(instance)
        data["query"] = json.loads(instance.query)
        return data


    class Meta:
        model = ApplyStemmerWorker
        fields = ("id", "url", "author_username", "indices", "description", "task", "query",)
