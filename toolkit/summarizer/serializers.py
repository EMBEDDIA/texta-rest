from rest_framework import serializers
from toolkit.serializer_constants import ProjectResourceUrlSerializer
from .models import Summarizer


class SummarizerSummarizeTextSerializer(serializers.ModelSerializer, ProjectResourceUrlSerializer):
    text = serializers.CharField(required=True, help_text='Text to summarize')

    class Meta:
        model = Summarizer
        fields = ('id',
                  'url',
                  'text')


class SummarizerSummarizeSerializer(serializers.Serializer):
    SUPPORTED_ALGORITHMS = (
        "textrank",
        "lexrank"
    )
    SUPPORTED_RATIOS = (
        "0",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
        "1"
    )
    text = serializers.ListField(child=serializers.CharField(), required=True)
    alogrithms = serializers.MultipleChoiceField(
        choices=SUPPORTED_ALGORITHMS,
        default=["lexrank"]
    )
    ratio = serializers.MultipleChoiceField(
        choices=SUPPORTED_RATIOS,
        default=["0.2"]
    )
