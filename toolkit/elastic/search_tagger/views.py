import json
from toolkit.view_constants import BulkDelete
from .serializers import SearchTaggerQuerySerializer, SearchTaggerFieldsSerializer
from rest_framework import permissions, viewsets
import rest_framework.filters as drf_filters
from django_filters import rest_framework as filters
from toolkit.permissions.project_permissions import ProjectResourceAllowed
from .models import SearchTagger
from django.db import transaction
from toolkit.core.project.models import Project
from toolkit.elastic.index.models import Index


class SearchTaggerQueryViewSet(viewsets.ModelViewSet, BulkDelete):
    serializer_class = SearchTaggerQuerySerializer
    filter_backends = (drf_filters.OrderingFilter, filters.DjangoFilterBackend)
    ordering_fields = (
    'id', 'author__username', 'description', 'fields', 'task__time_started', 'task__time_completed', 'f1_score',
    'precision', 'recall', 'task__status')
    permission_classes = (
        ProjectResourceAllowed,
        permissions.IsAuthenticated,
    )

    def get_queryset(self):
        return SearchTagger.objects.filter(project=self.kwargs['project_pk'])

    def perform_create(self, serializer):
        with transaction.atomic():
            project = Project.objects.get(id=self.kwargs['project_pk'])
            indices = [index["name"] for index in serializer.validated_data["indices"]]
            indices = project.get_available_or_all_project_indices(indices)
            serializer.validated_data.pop("indices")

            worker: SearchTagger = serializer.save(
                    author=self.request.user,
                    project=project,
                    fields=json.dumps(serializer.validated_data["fields"]),
                )
            for index in Index.objects.filter(name__in=indices, is_open=True):
                worker.indices.add(index)
            worker.process()


class SearchTaggerFieldsViewSet(viewsets.ModelViewSet, BulkDelete):
    serializer_class = SearchTaggerFieldsSerializer
    filter_backends = (drf_filters.OrderingFilter, filters.DjangoFilterBackend)
    ordering_fields = (
        'id', 'author__username', 'description', 'fields', 'task__time_started', 'task__time_completed', 'f1_score',
        'precision', 'recall', 'task__status')
    permission_classes = (
        ProjectResourceAllowed,
        permissions.IsAuthenticated,
    )

    def get_queryset(self):
        return SearchTagger.objects.filter(project=self.kwargs['project_pk'])

    def perform_create(self, serializer):
        with transaction.atomic():
            project = Project.objects.get(id=self.kwargs['project_pk'])
            indices = [index["name"] for index in serializer.validated_data["indices"]]
            indices = project.get_available_or_all_project_indices(indices)
            serializer.validated_data.pop("indices")

            worker: SearchTagger = serializer.save(
                author=self.request.user,
                project=project,
                fields=json.dumps(serializer.validated_data["fields"]),
            )
            for index in Index.objects.filter(name__in=indices, is_open=True):
                worker.indices.add(index)
            worker.process()
