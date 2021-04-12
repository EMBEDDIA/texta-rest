from rest_framework import serializers
from toolkit.serializer_constants import FieldParseSerializer


class SummarizerSummerizeTextSerializer(FieldParseSerializer, serializers.Serializer):
    text = serializers.CharField(required=True, help_text='Text to summarize')
