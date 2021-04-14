import rest_framework.filters as drf_filters
from django_filters import rest_framework as filters
from rest_framework import permissions, status, viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import action
from .serializers import SummarizerSummarizeTextSerializer
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
    pass


class SummarizerApplyToIndex(APIView):
    pass
