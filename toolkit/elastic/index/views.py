import elasticsearch_dsl
import rest_framework.filters as drf_filters
from django.db import transaction
from django.http import JsonResponse
from django_filters import rest_framework as filters
from rest_auth import views
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from toolkit.elastic.exceptions import ElasticIndexAlreadyExists
from toolkit.elastic.index.models import Index
from toolkit.elastic.index.serializers import (
    AddTextaFactsMapping,
    IndexBulkDeleteSerializer, IndexSerializer,
)
from toolkit.elastic.tools.core import ElasticCore
from toolkit.permissions.project_permissions import IsSuperUser
from toolkit.settings import TEXTA_TAGS_KEY


class IndicesFilter(filters.FilterSet):
    id = filters.CharFilter('id', lookup_expr='exact')
    name = filters.CharFilter('name', lookup_expr='icontains')
    is_open = filters.BooleanFilter("is_open")


    class Meta:
        model = Index
        fields = []


class ElasticGetIndices(views.APIView):
    permission_classes = (IsSuperUser,)


    def get(self, request):
        """
        Returns **all** available indices from Elasticsearch.
        This is different from get_indices action in project view as it lists **all** indices in Elasticsearch.
        """
        es_core = ElasticCore()
        es_core.syncher()
        indices = [index.name for index in Index.objects.all()]
        return JsonResponse(indices, safe=False, status=status.HTTP_200_OK)


class IndexViewSet(mixins.CreateModelMixin,
                   mixins.ListModelMixin,
                   mixins.RetrieveModelMixin,
                   mixins.DestroyModelMixin,
                   viewsets.GenericViewSet):
    queryset = Index.objects.all()
    serializer_class = IndexSerializer
    permission_classes = [IsSuperUser]

    filter_backends = (drf_filters.OrderingFilter, filters.DjangoFilterBackend)
    pagination_class = None
    filterset_class = IndicesFilter

    ordering_fields = (
        'id',
        'name',
        'is_open'
    )


    def _resolve_cluster_differences(self, mapping_dict: dict):
        # Trying to support ES6 and ES7 mapping structure.
        keys = list(mapping_dict.keys())
        mapping_key = keys[0] if keys else None
        if mapping_key:
            mapping_dict = mapping_dict if "properties" in mapping_dict else mapping_dict[mapping_key]
            return mapping_dict
        # In this case, the mapping is a a plain dictionary because no fields exist.
        else:
            return None


    def _check_for_facts(self, index_mappings: dict, index_name: str):
        m = elasticsearch_dsl.Mapping()
        mapping_dict = index_mappings[index_name]["mappings"]
        mapping_dict = self._resolve_cluster_differences(mapping_dict)
        if mapping_dict:
            try:
                m._update_from_dict(mapping_dict)
            except:
                print(index_name, mapping_dict)
        else:
            return False

        try:
            has_texta_facts = isinstance(m[TEXTA_TAGS_KEY], elasticsearch_dsl.Nested)
            return has_texta_facts
        except KeyError:
            return False


    def list(self, request, *args, **kwargs):
        ec = ElasticCore()
        ec.syncher()
        response = super(IndexViewSet, self).list(request, *args, **kwargs)

        data = response.data  # Get the paginated and sorted queryset results.
        open_indices = [index for index in data if index["is_open"]]
        mappings = ec.es.indices.get_mapping()

        # Doing a stats request with no indices causes trouble.
        if open_indices:
            stats = ec.get_index_stats()

            # Update the paginated and sorted queryset results.
            for index in response.data:
                name = index["name"]
                is_open = index["is_open"]
                if is_open:
                    has_texta_facts_mapping = self._check_for_facts(index_mappings=mappings, index_name=name)
                    index.update(**stats[name], has_validated_facts=has_texta_facts_mapping)
                else:
                    # For the sake of courtesy on the front-end, make closed indices values zero.
                    index.update(size=0, doc_count=0, has_validated_facts=False)

        return response


    def retrieve(self, request, *args, **kwargs):
        ec = ElasticCore()
        response = super(IndexViewSet, self).retrieve(*args, *kwargs)
        if response.data["is_open"]:
            index_name = response.data["name"]
            mapping = ec.es.indices.get_mapping(index_name)
            has_validated_facts = self._check_for_facts(mapping, index_name)
            stats = ec.get_index_stats()
            response.data.update(**stats[index_name], has_validated_facts=has_validated_facts)
        else:
            response.data.update(size=0, doc_count=0, has_validated_facts=False)

        return response


    def create(self, request, **kwargs):
        data = IndexSerializer(data=request.data)
        data.is_valid(raise_exception=True)

        es = ElasticCore()
        index = data.validated_data["name"]
        is_open = data.validated_data["is_open"]

        # Using get_or_create to avoid unique name constraints on creation.
        if es.check_if_indices_exist([index]):
            # Even if the index already exists, create the index object just in case
            index, is_created = Index.objects.get_or_create(name=index)
            if is_created:
                index.is_open = is_open
            index.save()
            raise ElasticIndexAlreadyExists()

        else:
            index, is_created = Index.objects.get_or_create(name=index)
            if is_created:
                index.is_open = is_open
            index.save()

            es.create_index(index=index)
            if not is_open:
                es.close_index(index)
            return Response({"message": f"Added index {index} into Elasticsearch!"}, status=status.HTTP_201_CREATED)


    def destroy(self, request, pk=None, **kwargs):
        with transaction.atomic():
            index_name = Index.objects.get(pk=pk).name
            es = ElasticCore()
            es.delete_index(index_name)
            Index.objects.filter(pk=pk).delete()
            return Response({"message": f"Deleted index {index_name} from Elasticsearch!"})


    @action(detail=False, methods=['post'], serializer_class=IndexBulkDeleteSerializer)
    def bulk_delete(self, request, project_pk=None):
        serializer: IndexBulkDeleteSerializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        # Initialize Elastic requirements.
        ec = ElasticCore()
        # Get the index names.
        ids = serializer.validated_data["ids"]
        objects = Index.objects.filter(pk__in=ids)
        index_names = [item.name for item in objects]
        # Ensure deletion on both Elastic and DB.
        if index_names:
            ec.delete_index(",".join(index_names))
        deleted = objects.delete()
        info = {"num_deleted": deleted[0], "deleted_types": deleted[1]}
        return Response(info, status=status.HTTP_200_OK)


    @action(detail=False, methods=['post'])
    def sync_indices(self, request, pk=None, project_pk=None):
        ElasticCore().syncher()
        return Response({"message": "Synched everything successfully!"}, status=status.HTTP_204_NO_CONTENT)


    @action(detail=True, methods=['patch'])
    def close_index(self, request, pk=None, project_pk=None):
        es_core = ElasticCore()
        index = Index.objects.get(pk=pk)
        es_core.close_index(index.name)
        index.is_open = False
        index.save()
        return Response({"message": f"Closed the index {index.name}"})


    @action(detail=True, methods=['patch'])
    def open_index(self, request, pk=None, project_pk=None):
        es_core = ElasticCore()
        index = Index.objects.get(pk=pk)
        es_core.open_index(index.name)
        if not index.is_open:
            index.is_open = True
            index.save()

        return Response({"message": f"Opened the index {index.name}"})


    @action(detail=True, methods=['post'], serializer_class=AddTextaFactsMapping)
    def add_facts_mapping(self, request, pk=None, project_pk=None):
        es_core = ElasticCore()
        index = Index.objects.get(pk=pk)
        if index.is_open:
            es_core.add_texta_facts_mapping(index.name)
            return Response({"message": f"Added the Texta Facts mapping for: {index.name}"})
        else:
            return Response({"message": f"Index {index.name} is closed, could not add the mapping!"}, status=status.HTTP_400_BAD_REQUEST)
