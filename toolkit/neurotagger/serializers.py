import json
import re

from rest_framework import serializers
from django.db.models import Avg

from . import choices
from .models import Neurotagger
from toolkit.constants import get_field_choices
from toolkit.core.task.serializers import TaskSerializer
from toolkit.settings import URL_PREFIX
from toolkit.serializer_constants import ProjectResourceUrlSerializer



class NeurotaggerSerializer(serializers.HyperlinkedModelSerializer, ProjectResourceUrlSerializer):
    fields = serializers.ListField(child=serializers.CharField(), help_text=f'Fields used to build the model.', write_only=True)
    fields_parsed = serializers.SerializerMethodField()
    fact_name = serializers.CharField(help_text=
        'Fact name used to train a multilabel model, with fact values as classes. If given, the "queries" field will be ignored.',
        required=False,
        allow_blank=True
    )

    queries = serializers.JSONField(help_text='JSON list of strings of Elasticsearch queries to train on', required=False)

    query_names = serializers.JSONField(help_text=
        "Label names for queries, if training on queries. If not given, in that case defaults to ['query_N']",
        required=False,
    )


    model_architecture = serializers.ChoiceField(choices=choices.model_arch_choices)
    seq_len = serializers.IntegerField(default=choices.DEFAULT_SEQ_LEN, help_text=f'Default: {choices.DEFAULT_SEQ_LEN}')
    vocab_size = serializers.IntegerField(default=choices.DEFAULT_VOCAB_SIZE, help_text=f'Default: {choices.DEFAULT_VOCAB_SIZE}')
    num_epochs = serializers.IntegerField(default=choices.DEFAULT_NUM_EPOCHS, help_text=f'Default: {choices.DEFAULT_NUM_EPOCHS}')
    validation_split = serializers.FloatField(default=choices.DEFAULT_VALIDATION_SPLIT, help_text=f'Default: {choices.DEFAULT_VALIDATION_SPLIT}')
    score_threshold = serializers.IntegerField(default=choices.DEFAULT_SCORE_THRESHOLD, help_text=f'Default: {choices.DEFAULT_SCORE_THRESHOLD}')

    negative_multiplier = serializers.FloatField(default=choices.DEFAULT_NEGATIVE_MULTIPLIER, help_text=f'Default: {choices.DEFAULT_NEGATIVE_MULTIPLIER}')
    maximum_sample_size = serializers.IntegerField(default=choices.DEFAULT_MAX_SAMPLE_SIZE,help_text=f'Default: {choices.DEFAULT_MAX_SAMPLE_SIZE}')
    minimum_fact_document_count = serializers.IntegerField(default=choices.DEFAULT_MIN_SAMPLE_SIZE, help_text=
    f'Minimum number of documents required per fact to train a multilabel model. If no fact name is chosen this option is ignored. Default: {choices.DEFAULT_MIN_SAMPLE_SIZE}')

    task = TaskSerializer(read_only=True)
    plot = serializers.SerializerMethodField()
    model_plot = serializers.SerializerMethodField()
    url = serializers.SerializerMethodField()


    class Meta:
        model = Neurotagger
        fields = ('url', 'id', 'description', 'project', 'author', 'queries', 'query_names', 'validation_split', 'score_threshold',
                  'fields', 'fields_parsed', 'model_architecture', 'seq_len', 'maximum_sample_size', 'negative_multiplier',
                  'location', 'num_epochs', 'vocab_size', 'plot', 'task', 'validation_accuracy', 'training_accuracy', 'fact_values',
                  'training_loss', 'validation_loss', 'model_plot', 'result_json', 'fact_name', 'minimum_fact_document_count',)

        read_only_fields = ('author', 'project', 'location', 'accuracy', 'loss', 'plot',
                            'model_plot', 'result_json', 'validation_accuracy', 'training_accuracy',
                            'training_loss', 'validation_loss', 'fact_values', 'classification_report'
                            )
        

    def __init__(self, *args, **kwargs):
        '''
        Add the ability to pass extra arguments such as "remove_fields".
        Useful for the Serializer eg in another Serializer, without making a new one.
        '''
        remove_fields = kwargs.pop('remove_fields', None)
        super(NeurotaggerSerializer, self).__init__(*args, **kwargs)

        if remove_fields:
            # for multiple fields in a list
            for field_name in remove_fields:
                self.fields.pop(field_name)
    

    def get_plot(self, obj):
        if obj.plot:
            return '{0}/{1}'.format(URL_PREFIX, obj.plot)
        else:
            return None

    def get_model_plot(self, obj):
        if obj.model_plot:
            return '{0}/{1}'.format(URL_PREFIX, obj.model_plot)
        else:
            return None

    def get_fields_parsed(self, obj):
        if obj.fields:
            return json.loads(obj.fields)
        return None
