from rest_framework import permissions, status, viewsets
from rest_framework.response import Response
from rest_framework.decorators import action
from .serializers import SummarizerSummerizeTextSerializer


class SummarizerViewSet(viewsets.ModelViewSet):
    ordering_fields = ()

    @action(detail=True, methods=['post'], serializer_class=SummarizerSummerizeTextSerializer)
    def summerize_text(self, request):
        serializer = SummarizerSummerizeTextSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        # summerize text
        result = ""
        return Response(result, status=status.HTTP_200_OK)