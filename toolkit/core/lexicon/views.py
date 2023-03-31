from django.shortcuts import render
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
import json

from toolkit.core.project.models import Project
from toolkit.core.lexicon.models import Lexicon
from toolkit.core.lexicon.serializers import LexiconSerializer, LexiconAddWordSerializer
from toolkit.permissions.project_permissions import ProjectAccessInApplicationsAllowed



class LexiconViewSet(viewsets.ModelViewSet):
    serializer_class = LexiconSerializer
    permission_classes = (
        ProjectAccessInApplicationsAllowed,
        permissions.IsAuthenticated,
        )


    def perform_create(self, serializer):
        serializer.save(author=self.request.user,
            project=Project.objects.get(id=self.kwargs['project_pk']),
                positives_used = json.dumps(serializer.validated_data.get('positives_used', [])),
                negatives_used = json.dumps(serializer.validated_data.get('negatives_used', [])),
                positives_unused = json.dumps(serializer.validated_data.get('positives_unused', [])),
                negatives_unused = json.dumps(serializer.validated_data.get('negatives_unused', [])))


    def perform_update(self, serializer):
        serializer.save(positives_used = json.dumps(serializer.validated_data.get('positives_used', [])),
                negatives_used = json.dumps(serializer.validated_data.get('negatives_used', [])),
                positives_unused = json.dumps(serializer.validated_data.get('positives_unused', [])),
                negatives_unused = json.dumps(serializer.validated_data.get('negatives_unused', [])))


    def get_queryset(self):
        return Lexicon.objects.filter(project=self.kwargs['project_pk']).order_by('-id')


    def create(self, request, *args, **kwargs):
        serializer = LexiconSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
    

    @action(detail=True, methods=["post"], serializer_class=LexiconAddWordSerializer)
    def add_positive_used(self, request, pk=None, project_pk=None):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        # get lexicon
        lexicon = Lexicon.objects.get(pk=pk)
        # load & append new item
        positives_used = json.loads(lexicon.positives_used)
        positives_used.append(serializer.validated_data["word"])
        # remove duplicates
        positives_used = list(set(positives_used))
        # dump & save new list
        lexicon.positives_used = json.dumps(positives_used)
        lexicon.save()
        return Response({"detail": "Success!"}, status=status.HTTP_200_OK)

