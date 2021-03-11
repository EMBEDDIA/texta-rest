import json
import os
import logging

import rest_framework.filters as drf_filters
from django.http import HttpResponse
from django_filters import rest_framework as filters
from django.db import transaction
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from toolkit.core.task.models import Task
from toolkit.core.project.models import Project

from toolkit.elastic.tools.core import ElasticCore
from toolkit.elastic.tools.feedback import Feedback
from toolkit.elastic.tools.searcher import ElasticSearcher

from toolkit.elastic.index.models import Index

from toolkit.exceptions import NonExistantModelError, ProjectValidationFailed, DownloadingModelsNotAllowedError, InvalidModelIdentifierError

from toolkit.permissions.project_permissions import ProjectResourceAllowed
from toolkit.serializer_constants import ProjectResourceImportModelSerializer
from toolkit.settings import CELERY_LONG_TERM_TASK_QUEUE, INFO_LOGGER
from toolkit.evaluator.models import Evaluator as EvaluatorObject
from toolkit.evaluator.serializers import EvaluatorSerializer
from toolkit.evaluator.tasks import evaluate_tags_task

from toolkit.view_constants import BulkDelete, FeedbackModelView


class EvaluatorFilter(filters.FilterSet):
    description = filters.CharFilter('description', lookup_expr='icontains')
    task_status = filters.CharFilter('task__status', lookup_expr='icontains')


    class Meta:
        model = EvaluatorObject
        fields = []


class EvaluatorViewSet(viewsets.ModelViewSet, BulkDelete, FeedbackModelView):
    serializer_class = EvaluatorSerializer
    permission_classes = (
        permissions.IsAuthenticated,
        ProjectResourceAllowed,
    )

    filter_backends = (drf_filters.OrderingFilter, filters.DjangoFilterBackend)
    filterset_class = EvaluatorFilter
    ordering_fields = ('id', 'author__username', 'description',  'task__time_started', 'task__time_completed', 'f1_score', 'precision', 'recall', 'task__status')


    def perform_create(self, serializer, **kwargs):
        project = Project.objects.get(id=self.kwargs['project_pk'])
        indices = [index["name"] for index in serializer.validated_data["indices"]]
        indices = project.get_available_or_all_project_indices(indices)

        serializer.validated_data.pop("indices")


        evaluator: EvaluatorObject = serializer.save(
            author=self.request.user,
            project=project,
            **kwargs
        )

        for index in Index.objects.filter(name__in=indices, is_open=True):
            evaluator.indices.add(index)

        query = serializer.validated_data["query"]

        new_task = Task.objects.create(evaluator=evaluator, status='created')
        evaluator.task = new_task
        evaluator.save()

        es_timeout = 10
        bulk_size = 100
        evaluate_tags_task.apply_async(args=(evaluator.pk, indices, query, es_timeout, bulk_size), queue=CELERY_LONG_TERM_TASK_QUEUE)


        #evaluator.evaluate_tags(indices, query)


    def get_queryset(self):
        return EvaluatorObject.objects.filter(project=self.kwargs['project_pk']).order_by('-id')



    @action(detail=True, methods=['get'])
    def export_model(self, request, pk=None, project_pk=None):
        zip_name = f'evaluator_model_{pk}.zip'

        tagger_object: EvaluatorObject = self.get_object()
        data = tagger_object.export_resources()
        response = HttpResponse(data)
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(zip_name)
        return response


    @action(detail=False, methods=["post"], serializer_class=ProjectResourceImportModelSerializer)
    def import_model(self, request, pk=None, project_pk=None):
        serializer = ProjectResourceImportModelSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        uploaded_file = serializer.validated_data['file']
        tagger_id = EvaluatorObject.import_resources(uploaded_file, request, project_pk)
        return Response({"id": tagger_id, "message": "Successfully imported model and associated files."}, status=status.HTTP_201_CREATED)
