from rest_framework import serializers
from toolkit.serializer_constants import ProjectResourceUrlSerializer
from .models import Summarizer
from .values import DefaultSummarizerValues
from toolkit.elastic.index.serializers import IndexSerializer


class SummarizerSummarizeTextSerializer(serializers.ModelSerializer, ProjectResourceUrlSerializer):
    indices = IndexSerializer(many=True, default=[])
    author_username = serializers.CharField(source='author.username', read_only=True, required=False)
    description = serializers.CharField()
    query = serializers.JSONField(help_text='Query in JSON format', required=False)
    fields = serializers.ListField(required=True)

    class Meta:
        model = Summarizer
        fields = ("id", "url", "author_username", "indices", "description", "query", "fields")


class SummarizerSummarizeSerializer(serializers.Serializer):
    text = serializers.ListField(child=serializers.CharField(), required=True)
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
