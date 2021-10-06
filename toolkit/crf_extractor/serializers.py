import json
from rest_framework import serializers, fields
from toolkit.core.user_profile.serializers import UserSerializer
from toolkit.elastic.tools.searcher import EMPTY_QUERY
from toolkit.serializer_constants import (
    FieldParseSerializer,
    IndicesSerializerMixin,
    ProjectResourceUrlSerializer,
    ProjectFilteredPrimaryKeyRelatedField
)
from toolkit.tagger.choices import DEFAULT_ES_TIMEOUT, DEFAULT_BULK_SIZE, DEFAULT_MAX_CHUNK_BYTES
from toolkit.embedding.models import Embedding
from .models import CRFExtractor
from .choices import FEATURE_FIELDS_CHOICES, FEATURE_EXTRACTOR_CHOICES


class CRFExtractorSerializer(serializers.ModelSerializer, IndicesSerializerMixin, ProjectResourceUrlSerializer):
    description = serializers.CharField(help_text='Description for the CRFExtractor model.')
    author = UserSerializer(read_only=True)
    query = serializers.JSONField(
        help_text='Filter the documents used in training. Default: All.',
        default=EMPTY_QUERY,
    )
    mlp_field = serializers.CharField(help_text='Text field used to build the model.')
    labels = serializers.JSONField(
        default=["GPE", "ORG", "PER", "LOC"],
        help_text="List of labels used to train the extraction model. Default: ['GPE', 'ORG', 'PER', 'LOC']"
    )
    num_iter = serializers.IntegerField(default=100)
    test_size = serializers.FloatField(
        default=0.3,
        help_text="Proportion of documents reserved for testing the model. Default: 0.3."
    )
    c1 = serializers.FloatField(default=1.0, help_text="Coefficient for L1 penalty. Default: 1.0.")
    c2 = serializers.FloatField(default=1.0, help_text="Coefficient for L2 penalty. Default: 1.0.")
    bias = serializers.BooleanField(
        default=True,
        help_text="Capture the proportion of a given label in the training set. Default: True"
    )
    window_size = serializers.IntegerField(
        default=2,
        help_text="Number of words before and after the observed word analyzed. Default: 2.",
        )
    suffix_len = serializers.CharField(
        default=json.dumps((2,2)),
        help_text="Number of characters (min, max) used for word suffixes as features. Default: (2, 2)"
        )
    feature_fields = fields.MultipleChoiceField(
        choices=FEATURE_FIELDS_CHOICES,
        default=FEATURE_FIELDS_CHOICES,
        help_text=f"Layers (MLP subfields) used as features for the observed word. Default: {FEATURE_FIELDS_CHOICES}"
    )
    context_feature_fields = fields.MultipleChoiceField(
        choices=FEATURE_FIELDS_CHOICES,
        default=FEATURE_FIELDS_CHOICES,
        help_text=f"Layers (MLP subfields) used as features for the context of the observed word. Default: {FEATURE_FIELDS_CHOICES}"
    )
    feature_extractors = fields.MultipleChoiceField(
        choices=FEATURE_EXTRACTOR_CHOICES,
        default=FEATURE_EXTRACTOR_CHOICES,
        help_text=f"Feature extractors used for the observed word and it's context. Default: {FEATURE_EXTRACTOR_CHOICES}"
    )
    embedding = ProjectFilteredPrimaryKeyRelatedField(
        queryset=Embedding.objects,
        many=False,
        read_only=False,
        allow_null=True,
        default=None,
        help_text="Embedding to use for finding similar words for the observed word and it's context. Default = None"
    )
    url = serializers.SerializerMethodField()


    class Meta:
        model = CRFExtractor
        fields = (
            'id', 'url', 'author', 'description', 'query', 'indices', 'mlp_field',
            'window_size', 'test_size', 'num_iter', 'c1', 'c2', 'bias', 'suffix_len',
            'labels', 'feature_fields', 'context_feature_fields', 'feature_extractors',
            'embedding'
        )
        read_only_fields = ()
        fields_to_parse = ('fields',)


class ApplyCRFExtractorSerializer(FieldParseSerializer, IndicesSerializerMixin):
    mlp_fields = serializers.ListField(child=serializers.CharField())
    query = serializers.JSONField(
        help_text='Filter the documents which to scroll and apply to.',
        default=EMPTY_QUERY
    )

    # should turn this into mixin!
    es_timeout = serializers.IntegerField(
        default=DEFAULT_ES_TIMEOUT,
        help_text=f"Elasticsearch scroll timeout in minutes. Default:{DEFAULT_ES_TIMEOUT}."
    )
    bulk_size = serializers.IntegerField(
        min_value=1,
        max_value=10000,
        default=DEFAULT_BULK_SIZE,
        help_text=f"How many documents should be sent towards Elasticsearch at once. Default:{DEFAULT_BULK_SIZE}."
    )
    max_chunk_bytes = serializers.IntegerField(
        min_value=1,
        default=DEFAULT_MAX_CHUNK_BYTES,
        help_text=f"Data size in bytes that Elasticsearch should accept to prevent Entity Too Large errors. Default:{DEFAULT_MAX_CHUNK_BYTES}."
    )


class CRFExtractorTagTextSerializer(serializers.Serializer):
    text = serializers.CharField(help_text="Use the CRF model on text.")
