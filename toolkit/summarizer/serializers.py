from rest_framework import serializers
from toolkit.serializer_constants import ProjectResourceUrlSerializer
from .models import Summarizer
from .values import DefaultSummarizerValues
from toolkit.elastic.index.serializers import IndexSerializer
from toolkit.settings import REST_FRAMEWORK
from django.urls import reverse


class SummarizerSummarizeSerializer(serializers.Serializer):
    text = serializers.CharField(required=True)
    algorithm = serializers.MultipleChoiceField(
        choices=DefaultSummarizerValues.SUPPORTED_ALGORITHMS,
        default=["lexrank"]
    )
    ratio = serializers.DecimalField(max_digits=2, decimal_places=1, default=0.2)


class SummarizerApplyToIndexSerializer(serializers.Serializer):
    indices = serializers.ListField(child=serializers.CharField(), required=True)
    fields = serializers.ListField(child=serializers.CharField(), required=True)
    query = serializers.JSONField(default="{}")
    algorithm = serializers.MultipleChoiceField(
        choices=DefaultSummarizerValues.SUPPORTED_ALGORITHMS,
        default=["lexrank"]
    )
    ratio = serializers.DecimalField(max_digits=2, decimal_places=1, default=0.2)


class SummarizerIndexSerializer(serializers.ModelSerializer):
    indices = IndexSerializer(many=True, default=[])
    author_username = serializers.CharField(source='author.username', read_only=True, required=False)
    description = serializers.CharField()
    url = serializers.SerializerMethodField()
    query = serializers.JSONField(help_text='Query in JSON format', required=False)
    fields = serializers.ListField(required=True)

    class Meta:
        model = Summarizer
        fields = ("id", "url", "author_username", "indices", "description", "query", "fields")

    def get_url(self, obj):
        default_version = REST_FRAMEWORK.get("DEFAULT_VERSION")
        index = reverse(f"{default_version}:summarizer_index-detail", kwargs={"project_pk": obj.project.pk, "pk": obj.pk})
        if "request" in self.context:
            request = self.context["request"]
            url = request.build_absolute_uri(index)
            return url
        else:
            return None
