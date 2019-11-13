from rest_framework import serializers
import json
import re

from toolkit.embedding.models import Embedding, Task, EmbeddingCluster
from toolkit.embedding import choices
from toolkit.core.task.serializers import TaskSerializer
from toolkit.serializer_constants import ProjectResourceUrlSerializer

class EmbeddingSerializer(serializers.HyperlinkedModelSerializer, ProjectResourceUrlSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)    
    task = TaskSerializer(read_only=True)
    fields = serializers.ListField(child=serializers.CharField(), help_text=f'Fields used to build the model.', write_only=True)
    num_dimensions = serializers.IntegerField(default=choices.DEFAULT_NUM_DIMENSIONS,
                                    help_text=f'Default: {choices.DEFAULT_NUM_DIMENSIONS}')
    min_freq = serializers.IntegerField(default=choices.DEFAULT_MIN_FREQ,
                                    help_text=f'Default: {choices.DEFAULT_MIN_FREQ}')
    max_documents = serializers.IntegerField(default=choices.DEFAULT_MAX_DOCUMENTS)
    query = serializers.JSONField(help_text='Query in JSON format', required=False)
    url = serializers.SerializerMethodField()
    fields_parsed = serializers.SerializerMethodField()

    class Meta:
        model = Embedding
        fields = ('id', 'author_username', 'url', 'description', 'fields', 'query', 'max_documents', 'num_dimensions', 'min_freq', 'vocab_size', 'task', 'fields_parsed')
        read_only_fields = ('vocab_size',)


    def get_fields_parsed(self, obj):
        if obj.fields:
            return json.loads(obj.fields)
        return None


class EmbeddingPredictSimilarWordsSerializer(serializers.Serializer):
    positives = serializers.ListField(child=serializers.CharField(), help_text=f'Positive words for the model.')
    negatives = serializers.ListField(child=serializers.CharField(), help_text=f'Negative words for the model. Default: EMPTY', required=False, default=[])
    output_size = serializers.IntegerField(default=choices.DEFAULT_OUTPUT_SIZE,
                                    help_text=f'Default: {choices.DEFAULT_OUTPUT_SIZE}')


class EmbeddingClusterSerializer(serializers.ModelSerializer, ProjectResourceUrlSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)    
    task = TaskSerializer(read_only=True)
    num_clusters = serializers.IntegerField(default=choices.DEFAULT_NUM_CLUSTERS, help_text=f'Default: {choices.DEFAULT_NUM_CLUSTERS}')
    description = serializers.CharField(default='', help_text=f'Default: EMPTY')
    vocab_size = serializers.SerializerMethodField()
    location = serializers.SerializerMethodField()
    url = serializers.SerializerMethodField()

    class Meta:
        model = EmbeddingCluster
        fields = ('id', 'author_username', 'url', 'description', 'embedding', 'vocab_size', 'num_clusters', 'location', 'task')

        read_only_fields = ('task',)

    def get_vocab_size(self, obj):
        return obj.embedding.vocab_size

    def get_location(self, obj):
        return json.loads(obj.location)

class EmbeddingClusterBrowserSerializer(serializers.Serializer):
    number_of_clusters = serializers.IntegerField(default=choices.DEFAULT_BROWSER_NUM_CLUSTERS, help_text=f'Default: {choices.DEFAULT_BROWSER_NUM_CLUSTERS}')
    max_examples_per_cluster = serializers.IntegerField(default=choices.DEFAULT_BROWSER_EXAMPLES_PER_CLUSTER, help_text=f'Default: {choices.DEFAULT_BROWSER_EXAMPLES_PER_CLUSTER}')
    cluster_order = serializers.ChoiceField(((False, 'ascending'), (True, 'descending')))
