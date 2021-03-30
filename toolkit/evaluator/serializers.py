import json
from typing import List

from rest_framework import serializers
from django.core.exceptions import ValidationError

from toolkit.core.project.models import Project
from toolkit.core.task.serializers import TaskSerializer

from toolkit.elastic.index.serializers import IndexSerializer
from toolkit.elastic.tools.searcher import EMPTY_QUERY

from toolkit.serializer_constants import FieldParseSerializer, ProjectResourceUrlSerializer

from toolkit.evaluator import choices
from toolkit.evaluator.models import Evaluator
from toolkit.evaluator.validators import (
    validate_fact,
    validate_fact_value,
    validate_metric_restrictions,
    validate_average_function,
    validate_fact_values_in_sync
)

class FilteredAverageSerializer(serializers.Serializer):
    min_count = serializers.IntegerField(default=choices.DEFAULT_MIN_COUNT, required=False, help_text=f"Required minimum number of tags present in the union set to include corresponding tag's scores to the average calculation. Default = {choices.DEFAULT_MIN_COUNT}.")
    max_count = serializers.IntegerField(default=choices.DEFAULT_MAX_COUNT, required=False, help_text=f"Required maximum number of tags present in the union set to include corresponding tag's scores to the average calculation. Default = {choices.DEFAULT_MAX_COUNT}.")
    metric_restrictions = serializers.JSONField(default={}, validators=[validate_metric_restrictions], required=False, help_text=f"Score restrictions in format {{metric: {{'min_score: min_score, 'max_score': max_score}}, ...}}. Default = No restrictions.")


class IndividualResultsSerializer(serializers.Serializer):
    min_count = serializers.IntegerField(default=choices.DEFAULT_MIN_COUNT, required=False, help_text=f"Required minimum number of tags present in the union set to include corresponding tag's scores to the output. Default = {choices.DEFAULT_MIN_COUNT}.")
    max_count = serializers.IntegerField(default=choices.DEFAULT_MAX_COUNT, required=False, help_text=f"Required maximum number of tags present in the union set to include corresponding tag's scores to the output. Default = {choices.DEFAULT_MAX_COUNT}.")
    metric_restrictions = serializers.JSONField(default={}, validators=[validate_metric_restrictions], required=False, help_text=f"Score restrictions in format {{metric: {{'min_score: min_score, 'max_score': max_score}}, ...}}. Default = No restrictions.")
    order_by = serializers.ChoiceField(default=choices.DEFAULT_ORDER_BY_FIELD, choices=choices.ORDERING_FIELDS_CHOICES, required=False, help_text=f"Field used for ordering the results. Default = {choices.DEFAULT_ORDER_BY_FIELD}")
    order_desc = serializers.BooleanField(default=choices.DEFAULT_ORDER_DESC, required=False, help_text=f"Order results in descending order? Default = {choices.DEFAULT_ORDER_DESC}.")


class EvaluatorSerializer(serializers.ModelSerializer, ProjectResourceUrlSerializer):
    author_username = serializers.CharField(source="author.username", read_only=True)
    query = serializers.JSONField(required=False, help_text="Query in JSON format", default=json.dumps(EMPTY_QUERY))
    indices = IndexSerializer(many=True, default=[])

    true_fact = serializers.CharField(required=True, help_text=f"Fact name used as true label for mulilabel evaluation.")
    predicted_fact = serializers.CharField(required=True, help_text=f"Fact name used as predicted label for multilabel evaluation.")

    true_fact_value = serializers.CharField(required=False, default="", help_text=f"Fact value used as true label for binary evaluation.")
    predicted_fact_value = serializers.CharField(required=False, default="", help_text=f"Fact value used as predicted label for binary evaluation.")

    average_function = serializers.ChoiceField(choices=choices.AVG_CHOICES, default=choices.DEFAULT_AVG_FUNCTION, required=False, help_text = f"Sklearn average function. Default = {choices.DEFAULT_AVG_FUNCTION}")

    es_timeout = serializers.IntegerField(default=choices.DEFAULT_ES_TIMEOUT, help_text=f"Elasticsearch scroll timeout in minutes. Default = {choices.DEFAULT_ES_TIMEOUT}.")
    scroll_size = serializers.IntegerField(min_value=1, max_value=10000, default=choices.DEFAULT_SCROLL_SIZE, help_text=f"How many documents should be returned by one Elasticsearch scroll. Default = {choices.DEFAULT_SCROLL_SIZE}.")

    add_individual_results = serializers.BooleanField(default=choices.DEFAULT_ADD_INDIVIDUAL_RESULTS, required=False, help_text=f"Only used for multilabel/multiclass evaluation. If enabled, individual label scores are calculated and stored as well. Default = {choices.DEFAULT_ADD_INDIVIDUAL_RESULTS}.")

    plot = serializers.SerializerMethodField()
    task = TaskSerializer(read_only=True)

    url = serializers.SerializerMethodField()


    def validate_indices(self, value):
        """ Check if indices exist in the relevant project. """
        project_obj = Project.objects.get(id=self.context["view"].kwargs["project_pk"])
        for index in value:
            if index.get("name") not in project_obj.get_indices():
                raise serializers.ValidationError(f'Index "{index.get("name")}" is not contained in your project indices "{project_obj.get_indices()}"')
        return value


    def validate(self, data):
        """ Check if all inserted facts and fact values are present in the indices."""
        indices = [index.get("name") for index in data.get("indices")]

        true_fact = data.get("true_fact")
        predicted_fact = data.get("predicted_fact")

        true_fact_value = data.get("true_fact_value")
        predicted_fact_value = data.get("predicted_fact_value")

        avg_function = data.get("average_function")

        validate_fact(indices, true_fact)
        validate_fact(indices, predicted_fact)

        validate_fact_value(indices, true_fact, true_fact_value)
        validate_fact_value(indices, predicted_fact, predicted_fact_value)

        validate_fact_values_in_sync(true_fact_value, predicted_fact_value)

        validate_average_function(avg_function, true_fact_value, predicted_fact_value)

        return data


    class Meta:
        model = Evaluator
        fields = ("url", "author_username", "id", "description", "indices", "query", "true_fact", "predicted_fact", "true_fact_value", "predicted_fact_value",
                  "average_function", "f1_score", "precision", "recall", "accuracy", "confusion_matrix", "n_true_classes", "n_predicted_classes", "n_total_classes",
                  "evaluation_type", "scroll_size", "es_timeout", "scores_imprecise", "score_after_scroll", "document_count", "add_individual_results", "plot", "task")

        read_only_fields = ("project", "f1_score", "precision", "recall", "accuracy", "confusion_matrix", "n_true_classes", "n_predicted_classes", "n_total_classes", "document_count", "evaluation_type", "scores_imprecise", "score_after_scroll","task")
