import rest_framework.filters as drf_filters
from django_filters import rest_framework as filters
from rest_framework import permissions, status, viewsets
from rest_framework.views import APIView
from rest_framework.renderers import BrowsableAPIRenderer, HTMLFormRenderer, JSONRenderer
from rest_framework.response import Response
from rest_framework.decorators import action
from .serializers import SummarizerSummarizeTextSerializer, SummarizerSummarizeSerializer, SummarizerApplyToIndexSerializer
from toolkit.permissions.project_permissions import ProjectResourceAllowed
from .models import Summarizer
from toolkit.view_constants import BulkDelete


class SummarizerViewSet(viewsets.ModelViewSet, BulkDelete):
    serializer_class = SummarizerSummarizeTextSerializer
    filter_backends = (drf_filters.OrderingFilter, filters.DjangoFilterBackend)
    permission_classes = (
        ProjectResourceAllowed,
        permissions.IsAuthenticated,
    )

    def get_queryset(self):
        return Summarizer.objects.filter(project=self.kwargs['project_pk'])

    @action(detail=True, methods=['post'], serializer_class=SummarizerSummarizeTextSerializer)
    def summarize_text(self, request):
        serializer = SummarizerSummarizeTextSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        # summarize text
        result = ""
        return Response(result, status=status.HTTP_200_OK)


class SummarizerSummarize(APIView):
    serializer_class = SummarizerSummarizeSerializer
    renderer_classes = (BrowsableAPIRenderer, JSONRenderer, HTMLFormRenderer)
    permission_classes = (permissions.IsAuthenticated,)

    def post(self, request):
        serializer = SummarizerSummarizeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        text = list(serializer.validated_data["text"])
        algorithm = list(serializer.validated_data["algorithm"])
        ratio = list(serializer.validated_data["ratio"])

        return Response({
            'text': text,
            'algorithm': algorithm,
            'ratio': ratio})


class SummarizerApplyToIndex(APIView):
    serializer_class = SummarizerApplyToIndexSerializer
    renderer_classes = (BrowsableAPIRenderer, JSONRenderer, HTMLFormRenderer)
    permission_classes = (permissions.IsAuthenticated,)

    def post(self, request):
        serializer = SummarizerApplyToIndexSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        indices = list(serializer.validated_data["indices"])
        fields = list(serializer.validated_data["fields"])
        query = ""
        algorithm = list(serializer.validated_data["algorithm"])
        ratio = list(serializer.validated_data["ratio"])

        return Response({
            'indices': indices,
            'fields': fields,
            'query': query,
            'algorithm': algorithm,
            'ratio': ratio})
