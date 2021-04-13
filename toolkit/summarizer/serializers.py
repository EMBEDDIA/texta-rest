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