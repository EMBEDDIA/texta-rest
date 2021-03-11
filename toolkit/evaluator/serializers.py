import json

from rest_framework import serializers
from toolkit.core.task.serializers import TaskSerializer
from toolkit.elastic.index.serializers import IndexSerializer
from toolkit.elastic.tools.searcher import EMPTY_QUERY
from toolkit.serializer_constants import FieldParseSerializer, ProjectResourceUrlSerializer
from toolkit.evaluator import choices
from toolkit.evaluator.models import Evaluator


class EvaluatorSerializer(serializers.ModelSerializer, ProjectResourceUrlSerializer):

    author_username = serializers.CharField(source='author.username', read_only=True)
    query = serializers.JSONField(required=False, help_text='Query in JSON format', default=EMPTY_QUERY)
    indices = IndexSerializer(many=True, default=[])

    true_fact = serializers.CharField(required=True, help_text=f'Fact name used as true label for mulilabel evaluation.')
    predicted_fact = serializers.CharField(required=True, help_text=f'Fact name used as predicted label for multilabel evaluation.')

    true_fact_value = serializers.CharField(required=False, default = "",  help_text=f'Fact value used as true label for binary evaluation.')
    predicted_fact_value = serializers.CharField(required=False, default = "", help_text=f'Fact value used as predicted label for binary evaluation.')

    average_function = serializers.ChoiceField(choices=choices.AVG_CHOICES, default=choices.DEFAULT_AVG_FUNCTION, required=False, help_text = f'Sklearn average function.')

    task = TaskSerializer(read_only=True)
    #plot = serializers.SerializerMethodField()
    url = serializers.SerializerMethodField()


    class Meta:
        model = Evaluator
        fields = ('url', 'author_username', 'id', 'description', 'indices', 'query', 'true_fact', 'predicted_fact', 'true_fact_value', 'predicted_fact_value',
                  'average_function', 'f1_score', 'precision', 'recall', 'accuracy', 'confusion_matrix', 'task')

        read_only_fields = ('project', 'f1_score', 'precision', 'recall', 'accuracy', 'confusion_matrix', 'task')
