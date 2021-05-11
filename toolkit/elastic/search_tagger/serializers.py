from .models import SearchQueryTagger, SearchFieldsTagger
from rest_framework import serializers
from toolkit.core.task.serializers import TaskSerializer
from toolkit.elastic.index.serializers import IndexSerializer
from toolkit.serializer_constants import FieldParseSerializer
from toolkit.settings import REST_FRAMEWORK
from django.urls import reverse


class SearchQueryTaggerSerializer(FieldParseSerializer, serializers.ModelSerializer):
    indices = IndexSerializer(many=True, default=[])
    author_username = serializers.CharField(source='author.username', read_only=True, required=False)
    description = serializers.CharField()
    task = TaskSerializer(read_only=True, required=False)
    url = serializers.SerializerMethodField()
    query = serializers.JSONField(help_text='Query in JSON format', required=False)
    mapping_field = serializers.CharField(required=True)
    fact_name = serializers.CharField()
    fact_value = serializers.CharField()

    class Meta:
        model = SearchQueryTagger
        fields = ("id", "url", "author_username", "indices", "description", "task", "query", "mapping_field", "fact_name", "fact_value")
        fields_to_parse = ['fields']

    def get_url(self, obj):
        default_version = REST_FRAMEWORK.get("DEFAULT_VERSION")
        index = reverse(f"{default_version}:search_query_tagger-detail", kwargs={"project_pk": obj.project.pk, "pk": obj.pk})
        if "request" in self.context:
            request = self.context["request"]
            url = request.build_absolute_uri(index)
            return url
        else:
            return None


class SearchFieldsTaggerSerializer(FieldParseSerializer, serializers.ModelSerializer):
    indices = IndexSerializer(many=True, default=[])
    author_username = serializers.CharField(source='author.username', read_only=True, required=False)
    description = serializers.CharField()
    task = TaskSerializer(read_only=True, required=False)
    url = serializers.SerializerMethodField()
    query = serializers.JSONField(help_text='Query in JSON format', required=False)
    fields = serializers.ListField(child=serializers.CharField(), required=True)
    fact_name = serializers.CharField()

    class Meta:
        model = SearchFieldsTagger
        fields = ("id", "url", "author_username", "indices", "description", "task", "query", "fields", "fact_name")
        fields_to_parse = ['fields']
