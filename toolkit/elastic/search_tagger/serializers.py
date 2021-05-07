from .models import SearchTagger
from rest_framework import serializers
from toolkit.core.task.serializers import TaskSerializer
from toolkit.elastic.index.serializers import IndexSerializer
from toolkit.serializer_constants import FieldParseSerializer


class SearchTaggerQuerySerializer(FieldParseSerializer, serializers.ModelSerializer):
    indices = IndexSerializer(many=True, default=[])
    author_username = serializers.CharField(source='author.username', read_only=True, required=False)
    description = serializers.CharField()
    task = TaskSerializer(read_only=True, required=False)
    url = serializers.SerializerMethodField()
    query = serializers.JSONField(help_text='Query in JSON format', required=False)
    fields = serializers.ListField(child=serializers.CharField(), required=True)

    class Meta:
        model = SearchTagger
        fields = ("id", "url", "author_username", "indices", "description", "task", "query", "fields")
        fields_to_parse = ['fields']


class SearchTaggerFieldsSerializer(FieldParseSerializer, serializers.ModelSerializer):
    indices = IndexSerializer(many=True, default=[])
    author_username = serializers.CharField(source='author.username', read_only=True, required=False)
    description = serializers.CharField()
    task = TaskSerializer(read_only=True, required=False)
    url = serializers.SerializerMethodField()
    query = serializers.JSONField(help_text='Query in JSON format', required=False)
    fields = serializers.ListField(child=serializers.CharField(), required=True)

    class Meta:
        model = SearchTagger
        fields = ("id", "url", "author_username", "indices", "description", "task", "query", "fields")
        fields_to_parse = ['fields']
