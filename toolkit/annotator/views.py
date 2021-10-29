import rest_framework.filters as drf_filters
from django_filters import rest_framework as filters
from rest_framework import mixins, permissions, status, viewsets
# Create your views here.
from rest_framework.decorators import action
from rest_framework.response import Response

from toolkit.annotator.models import Annotator, Labelset
from toolkit.annotator.serializers import AnnotatorSerializer, BinaryAnnotationSerializer, CommentSerializer, DocumentIDSerializer, EntityAnnotationSerializer, LabelsetSerializer, MultilabelAnnotationSerializer, ValidateDocumentSerializer
from toolkit.permissions.project_permissions import ProjectAccessInApplicationsAllowed
from toolkit.serializer_constants import EmptySerializer
from toolkit.view_constants import BulkDelete


class LabelsetViewset(mixins.CreateModelMixin,
                      mixins.ListModelMixin,
                      mixins.RetrieveModelMixin,
                      mixins.DestroyModelMixin,
                      viewsets.GenericViewSet,
                      BulkDelete):
    serializer_class = LabelsetSerializer
    permission_classes = (
        ProjectAccessInApplicationsAllowed,
        permissions.IsAuthenticated,
    )

    filter_backends = (drf_filters.OrderingFilter, filters.DjangoFilterBackend)


    def get_queryset(self):
        return Labelset.objects.filter().order_by('-id')


class AnnotatorViewset(mixins.CreateModelMixin,
                       mixins.ListModelMixin,
                       mixins.RetrieveModelMixin,
                       mixins.DestroyModelMixin,
                       viewsets.GenericViewSet,
                       BulkDelete):
    serializer_class = AnnotatorSerializer
    permission_classes = (
        ProjectAccessInApplicationsAllowed,
        permissions.IsAuthenticated,
    )

    filter_backends = (drf_filters.OrderingFilter, filters.DjangoFilterBackend)


    @action(detail=True, methods=["POST"], serializer_class=EmptySerializer)
    def pull_document(self, request, pk=None, project_pk=None):
        annotator: Annotator = self.get_object()
        document = annotator.pull_document()
        if document:
            return Response(document)
        else:
            return Response({"detail": "No more documents left!"}, status=status.HTTP_404_NOT_FOUND)


    @action(detail=True, methods=["POST"], serializer_class=EmptySerializer)
    def pull_annotated(self, request, pk=None, project_pk=None):
        annotator: Annotator = self.get_object()
        document = annotator.pull_annotated_document()
        if document:
            return Response(document)
        else:
            return Response({"detail": "No more documents left!"}, status=status.HTTP_404_NOT_FOUND)


    @action(detail=True, methods=["POST"], serializer_class=DocumentIDSerializer)
    def skip_document(self, request, pk=None, project_pk=None):
        serializer: DocumentIDSerializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        annotator: Annotator = self.get_object()
        annotator.skip_document(serializer.validated_data["document_id"])
        return Response({"detail": f"Skipped document with ID: {serializer.validated_data['document_id']}"})


    @action(detail=True, methods=["POST"], serializer_class=ValidateDocumentSerializer)
    def validate_document(self, request, pk=None, project_pk=None):
        serializer: ValidateDocumentSerializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        annotator: Annotator = self.get_object()
        annotator.validate_document(
            document_id=serializer.validated_data["document_id"],
            facts=serializer.validated_data["facts"],
            is_valid=serializer.validated_data["is_valid"]
        )
        return Response({"detail": f"Validated document with ID: {serializer.validated_data['document_id']}"})


    @action(detail=True, methods=["POST"], serializer_class=EntityAnnotationSerializer)
    def annotate_entity(self, request, pk=None, project_pk=None):
        serializer: EntityAnnotationSerializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        annotator: Annotator = self.get_object()
        annotator.add_entity(
            document_id=serializer.validated_data["document_id"],
            fact_name=serializer.validated_data["fact_name"],
            fact_value=serializer.validated_data["fact_value"],
            spans=serializer.validated_data["spans"]
        )
        return Response({"detail": f"Skipped document with ID: {serializer.validated_data['document_id']}"})


    @action(detail=True, methods=["POST"], serializer_class=BinaryAnnotationSerializer)
    def annotate_binary(self, request, pk=None, project_pk=None):
        serializer: BinaryAnnotationSerializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        annotator: Annotator = self.get_object()
        choice = serializer.validated_data["annotation_type"]

        if choice == "pos":
            annotator.add_pos_label(serializer.validated_data["document_id"])
            return Response({"detail": f"Annotated document with ID: {serializer.validated_data['document_id']} with the pos label '{annotator.binary_configuration.pos_value}'"})

        elif choice == "neg":
            annotator.add_neg_label(serializer.validated_data["document_id"])
            return Response({"detail": f"Annotated document with ID: {serializer.validated_data['document_id']} with the neg label '{annotator.binary_configuration.neg_value}'"})


    @action(detail=True, methods=["POST"], serializer_class=MultilabelAnnotationSerializer)
    def annotate_multilabel(self, request, pk=None, project_pk=None):
        serializer: MultilabelAnnotationSerializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        annotator: Annotator = self.get_object()
        annotator.add_labels(serializer.validated_data["document_id"], serializer.validated_data["labels"])
        return Response({"detail": f"Annotated document with ID: {serializer.validated_data['document_id']} with the neg label '{annotator.binary_configuration.neg_value}'"})


    @action(detail=True, methods=["POST"], serializer_class=CommentSerializer)
    def add_comment(self, request, pk=None, project_pk=None):
        serializer: CommentSerializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        annotator: Annotator = self.get_object()
        annotator.add_comment(document_id=serializer.validated_data["document_id"], comment=serializer.validated_data["text"], user=request.user)
        return Response({"detail": f"Annotated document with ID: {serializer.validated_data['document_id']} with the neg label '{annotator.binary_configuration.neg_value}'"})


    def get_queryset(self):
        return Annotator.objects.filter(project=self.kwargs['project_pk']).order_by('-id')