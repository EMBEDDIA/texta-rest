from rest_framework import serializers
from toolkit.serializer_constants import ProjectResourceUrlSerializer
from .models import Summarizer
from .values import DefaultSummarizerValues


class SummarizerSummarizeTextSerializer(serializers.ModelSerializer, ProjectResourceUrlSerializer):
    text = serializers.CharField(required=True, help_text='Text to summarize')

    class Meta:
        model = Summarizer
        fields = ('id',
                  'url',
                  'text')


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
