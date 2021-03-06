import json
import logging
import os
import re

import rest_framework.filters as drf_filters
from celery import group
from celery.result import allow_join_result
from django.http import HttpResponse
from django_filters import rest_framework as filters
from rest_framework import mixins, permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response

from toolkit.core.health.utils import get_redis_status
from toolkit.core.project.models import Project
from toolkit.core.task.models import Task
from toolkit.elastic.core import ElasticCore
from toolkit.elastic.query import Query
from toolkit.elastic.searcher import ElasticSearcher
from toolkit.exceptions import NonExistantModelError, RedisNotAvailable, SerializerNotValid
from toolkit.helper_functions import add_finite_url_to_feedback
from toolkit.mlp.tasks import apply_mlp_on_list
from toolkit.permissions.project_permissions import ProjectResourceAllowed
from toolkit.serializer_constants import ProjectResourceImportModelSerializer
from toolkit.settings import CELERY_LONG_TERM_TASK_QUEUE, CELERY_MLP_TASK_QUEUE, CELERY_SHORT_TERM_TASK_QUEUE, INFO_LOGGER
from toolkit.tagger.models import TaggerGroup
from toolkit.tagger.serializers import TagRandomDocSerializer, TaggerGroupSerializer, TaggerGroupTagDocumentSerializer, TaggerGroupTagTextSerializer
from toolkit.tagger.tasks import apply_tagger, create_tagger_objects, save_tagger_results, start_tagger_task, train_tagger_task
from toolkit.tagger.validators import validate_input_document
from toolkit.view_constants import BulkDelete, TagLogicViews


class TaggerGroupFilter(filters.FilterSet):
    description = filters.CharFilter('description', lookup_expr='icontains')


    class Meta:
        model = TaggerGroup
        fields = []


class TaggerGroupViewSet(mixins.CreateModelMixin,
                         mixins.ListModelMixin,
                         mixins.RetrieveModelMixin,
                         mixins.DestroyModelMixin,
                         mixins.UpdateModelMixin,
                         viewsets.GenericViewSet,
                         TagLogicViews,
                         BulkDelete):
    queryset = TaggerGroup.objects.all()
    serializer_class = TaggerGroupSerializer
    permission_classes = (
        permissions.IsAuthenticated,
        ProjectResourceAllowed,
    )

    filter_backends = (drf_filters.OrderingFilter, filters.DjangoFilterBackend)
    filterset_class = TaggerGroupFilter
    ordering_fields = ('id', 'author__username', 'description', 'fact_name', 'minimum_sample_size', 'num_tags')


    def get_queryset(self):
        return TaggerGroup.objects.filter(project=self.kwargs['project_pk'])


    def create(self, request, *args, **kwargs):
        # add dummy value to tagger so serializer is happy
        request_data = request.data.copy()
        request_data.update({'tagger.description': 'dummy value'})

        # validate serializer again with updated values
        serializer = TaggerGroupSerializer(data=request_data, context={'request': request, 'view': self})
        serializer.is_valid(raise_exception=True)

        fact_name = serializer.validated_data['fact_name']
        active_project = Project.objects.get(id=self.kwargs['project_pk'])
        serialized_indices = [index["name"] for index in serializer.validated_data["tagger"]["indices"]]
        indices = Project.objects.get(pk=kwargs["project_pk"]).get_available_or_all_project_indices(serialized_indices)
        if not indices:
            raise ValidationError("No indices are available to you!")

        # retrieve tags with sufficient counts & create queries to build models
        tags = self.get_tags(fact_name, active_project, min_count=serializer.validated_data['minimum_sample_size'], indices=indices)

        # check if found any tags to build models on
        if not tags:
            return Response({'error': f'found no tags for fact name: {fact_name}'}, status=status.HTTP_400_BAD_REQUEST)

        # create queries for taggers
        tag_queries = self.create_queries(fact_name, tags)

        # retrive tagger options from hybrid tagger serializer
        validated_tagger_data = serializer.validated_data.pop('tagger')
        validated_tagger_data.update('')

        # create tagger group object
        tagger_group = serializer.save(
            author=self.request.user,
            project=Project.objects.get(id=self.kwargs['project_pk']),
            num_tags=len(tags)
        )

        # create taggers objects inside tagger group object
        # use async to make things faster
        create_tagger_objects.apply_async(args=(tagger_group.pk, validated_tagger_data, tags, tag_queries), queue=CELERY_LONG_TERM_TASK_QUEUE)

        # retrieve headers and create response
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        tagger_objects = instance.taggers.all()
        for tagger in tagger_objects:
            self.perform_destroy(tagger)

        self.perform_destroy(instance)

        return Response({"success": "Taggergroup instance deleted, related tagger instances deleted and related models and plots removed"}, status=status.HTTP_204_NO_CONTENT)


    @action(detail=True, methods=['get'])
    def models_list(self, request, pk=None, project_pk=None):
        """
        API endpoint for listing tagger objects connected to tagger group instance.
        """
        path = re.sub(r'tagger_groups/\d+/models_list/*$', 'taggers/', request.path)
        tagger_url_prefix = request.build_absolute_uri(path)
        tagger_objects = TaggerGroup.objects.get(id=pk).taggers.all()
        response = [{'tag': tagger.description, 'id': tagger.id, 'url': f'{tagger_url_prefix}{tagger.id}/', 'status': tagger.task.status} for tagger in tagger_objects]

        return Response(response, status=status.HTTP_200_OK)


    @action(detail=True, methods=['get'])
    def export_model(self, request, pk=None, project_pk=None):
        """Returns list of tags for input text."""
        zip_name = f'tagger_group_{pk}.zip'

        tagger_object: TaggerGroup = self.get_object()
        data = tagger_object.export_resources()
        response = HttpResponse(data)
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(zip_name)
        return response


    @action(detail=False, methods=["post"], serializer_class=ProjectResourceImportModelSerializer)
    def import_model(self, request, pk=None, project_pk=None):
        serializer = ProjectResourceImportModelSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        uploaded_file = serializer.validated_data['file']
        tagger_id = TaggerGroup.import_resources(uploaded_file, request, project_pk)
        return Response({"id": tagger_id, "message": "Successfully imported TaggerGroup models and associated files."}, status=status.HTTP_201_CREATED)


    @action(detail=True, methods=['post'])
    def models_retrain(self, request, pk=None, project_pk=None):
        """
        API endpoint for retraining tagger model.
        """
        instance = self.get_object()
        chain = start_tagger_task.s() | train_tagger_task.s() | save_tagger_results.s()

        # start retraining tasks
        for tagger in instance.taggers.all():
            # update task status so statistics are correct during retraining
            tagger.status = Task.STATUS_CREATED
            tagger.save()
            chain.apply_async(args=(tagger.pk,), queue=CELERY_LONG_TERM_TASK_QUEUE)

        return Response({'success': 'retraining tasks created', 'tagger_group_id': instance.id}, status=status.HTTP_200_OK)


    def get_mlp(self, text, lemmatize=False, use_ner=True):
        """
        Retrieves lemmas.
        Retrieves tags predicted by MLP NER and present in models.
        :return: string, list
        """
        tags = []
        hybrid_tagger_object = self.get_object()
        taggers = {t.description.lower(): {"tag": t.description, "id": t.id} for t in hybrid_tagger_object.taggers.all()}

        if lemmatize or use_ner:
            logging.getLogger(INFO_LOGGER).info(f"[Get MLP] Applying lemmatization and NER...")
            with allow_join_result():
                mlp = apply_mlp_on_list.apply_async(kwargs={"texts": [text], "analyzers": ["all"]}, queue=CELERY_MLP_TASK_QUEUE).get()
                mlp_result = mlp[0]
                logging.getLogger(INFO_LOGGER).info(f"[Get MLP] Finished applying MLP.")

        # lemmatize
        if lemmatize and mlp_result:
            text = mlp_result["text"]["lemmas"]
            lemmas_exists = True if text.strip() else False
            logging.getLogger(INFO_LOGGER).info(f"[Get MLP] Lemmatization result exists: {lemmas_exists}")
        # retrieve tags
        if use_ner and mlp_result:
            seen_tags = {}
            for fact in mlp_result["texta_facts"]:
                fact_val = fact["str_val"].lower().strip()
                if fact_val in taggers and fact_val not in seen_tags:
                    fact_val_dict = {
                        "tag": taggers[fact_val]["tag"],
                        "probability": 1.0,
                        "tagger_id": taggers[fact_val]["id"],
                        "ner_match": True
                    }
                    tags.append(fact_val_dict)
                    seen_tags[fact_val] = True
            logging.getLogger(INFO_LOGGER).info(f"[Get MLP] Detected {len(tags)} with NER.")
        return text, tags


    def get_tag_candidates(self, text, ignore_tags=[], n_similar_docs=10, max_candidates=10):
        """
        Finds frequent tags from documents similar to input document.
        Returns empty list if hybrid option false.
        """
        hybrid_tagger_object = self.get_object()
        field_paths = json.loads(hybrid_tagger_object.taggers.first().fields)
        indices = hybrid_tagger_object.project.get_indices()
        logging.getLogger(INFO_LOGGER).info(f"[Get Tag Candidates] Selecting from following indices: {indices}.")
        ignore_tags = {tag["tag"]: True for tag in ignore_tags}
        # create query
        query = Query()
        query.add_mlt(field_paths, text)
        # create Searcher object for MLT
        es_s = ElasticSearcher(indices=indices, query=query.query)
        logging.getLogger(INFO_LOGGER).info(f"[Get Tag Candidates] Trying to retrieve {n_similar_docs} documents from Elastic...")
        docs = es_s.search(size=n_similar_docs)
        logging.getLogger(INFO_LOGGER).info(f"[Get Tag Candidates] Successfully retrieved {len(docs)} documents from Elastic.")
        # dict for tag candidates from elastic
        tag_candidates = {}
        # retrieve tags from elastic response
        for doc in docs:
            if "texta_facts" in doc:
                for fact in doc["texta_facts"]:
                    if fact["fact"] == hybrid_tagger_object.fact_name:
                        fact_val = fact["str_val"]
                        if fact_val not in ignore_tags:
                            if fact_val not in tag_candidates:
                                tag_candidates[fact_val] = 0
                            tag_candidates[fact_val] += 1
        # sort and limit candidates
        tag_candidates = [item[0] for item in sorted(tag_candidates.items(), key=lambda k: k[1], reverse=True)][:max_candidates]
        logging.getLogger(INFO_LOGGER).info(f"[Get Tag Candidates] Retrieved {len(tag_candidates)} tag candidates.")
        return tag_candidates


    @action(detail=True, methods=['post'], serializer_class=TaggerGroupTagTextSerializer)
    def tag_text(self, request, pk=None, project_pk=None):
        """
        API endpoint for tagging raw text with tagger group.
        """
        logging.getLogger(INFO_LOGGER).info(f"[Tag Text] Starting tag_text...")
        data = request.data
        serializer = TaggerGroupTagTextSerializer(data=data)
        # check if valid request
        if not serializer.is_valid():
            raise SerializerNotValid(detail=serializer.errors)
        hybrid_tagger_object = self.get_object()
        # check if any of the models ready
        if not hybrid_tagger_object.taggers.filter(task__status=Task.STATUS_COMPLETED):
            raise NonExistantModelError()
        # error if redis not available
        if not get_redis_status()['alive']:
            raise RedisNotAvailable()
        # declare tag candidates variables
        text = serializer.validated_data['text']
        n_similar_docs = serializer.validated_data['n_similar_docs']
        n_candidate_tags = serializer.validated_data['n_candidate_tags']
        lemmatize = serializer.validated_data['lemmatize']
        use_ner = serializer.validated_data['use_ner']
        feedback = serializer.validated_data['feedback_enabled']
        # update text and tags with MLP
        text, tags = self.get_mlp(text, lemmatize=lemmatize, use_ner=use_ner)
        # retrieve tag candidates
        tag_candidates = self.get_tag_candidates(text, ignore_tags=tags, n_similar_docs=n_similar_docs, max_candidates=n_candidate_tags)
        # get tags
        tags += self.apply_tagger_group(text, tag_candidates, request, input_type='text', feedback=feedback)
        return Response(tags, status=status.HTTP_200_OK)


    @action(detail=True, methods=['post'], serializer_class=TaggerGroupTagDocumentSerializer)
    def tag_doc(self, request, pk=None, project_pk=None):
        """
        API endpoint for tagging JSON documents with tagger group.
        """
        logging.getLogger(INFO_LOGGER).info(f"[Tag Doc] Starting tag_doc...")
        data = request.data
        serializer = TaggerGroupTagDocumentSerializer(data=data)
        # check if valid request
        if not serializer.is_valid():
            raise SerializerNotValid(detail=serializer.errors)
        hybrid_tagger_object = self.get_object()
        # check if any of the models ready
        if not hybrid_tagger_object.taggers.filter(task__status=Task.STATUS_COMPLETED):
            raise NonExistantModelError()
        # error if redis not available
        if not get_redis_status()['alive']:
            raise RedisNotAvailable('Redis not available. Check if Redis is running.')
        # retrieve field data from the first element
        # we can do that safely because all taggers inside
        # hybrid tagger instance are trained on same fields
        hybrid_tagger_field_data = json.loads(hybrid_tagger_object.taggers.first().fields)
        # declare input_document variable
        input_document = serializer.validated_data['doc']
        # validate input document
        input_document = validate_input_document(input_document, hybrid_tagger_field_data)
        if isinstance(input_document, Exception):
            return input_document
        # combine document field values into one string
        combined_texts = '\n'.join(input_document.values())

        # declare tag candidates variables
        n_similar_docs = serializer.validated_data['n_similar_docs']
        n_candidate_tags = serializer.validated_data['n_candidate_tags']
        lemmatize = serializer.validated_data['lemmatize']
        use_ner = serializer.validated_data['use_ner']
        feedback = serializer.validated_data['feedback_enabled']
        # update text and tags with MLP
        combined_texts, tags = self.get_mlp(combined_texts, lemmatize=lemmatize, use_ner=use_ner)
        # retrieve tag candidates
        tag_candidates = self.get_tag_candidates(combined_texts, ignore_tags=tags, n_similar_docs=n_similar_docs, max_candidates=n_candidate_tags)
        # get tags
        tags += self.apply_tagger_group(input_document, tag_candidates, request, input_type='doc', lemmatize=lemmatize, feedback=feedback)
        return Response(tags, status=status.HTTP_200_OK)


    @action(detail=True, methods=['post'])
    def tag_random_doc(self, request, pk=None, project_pk=None):
        """
        API endpoint for tagging a random document.
        """
        logging.getLogger(INFO_LOGGER).info(f"[Tag Random doc] Starting tag_random_doc...")
        # get hybrid tagger object
        hybrid_tagger_object = self.get_object()

        # check if any of the models ready
        if not hybrid_tagger_object.taggers.filter(task__status=Task.STATUS_COMPLETED):
            raise NonExistantModelError()

        # retrieve tagger fields from the first object
        tagger_fields = json.loads(hybrid_tagger_object.taggers.first().fields)
        # error if redis not available

        if not get_redis_status()['alive']:
            raise RedisNotAvailable('Redis not available. Check if Redis is running.')

        serializer = TagRandomDocSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        project_object = Project.objects.get(pk=project_pk)
        indices = [index["name"] for index in serializer.validated_data["indices"]]
        indices = project_object.get_available_or_all_project_indices(indices)

        if not ElasticCore().check_if_indices_exist(indices):
            return Response({'error': f'One or more index from {list(hybrid_tagger_object.project.get_indices())} does not exist'}, status=status.HTTP_400_BAD_REQUEST)

        # retrieve random document
        random_doc = ElasticSearcher(indices=indices).random_documents(size=1)[0]
        # filter out correct fields from the document
        random_doc_filtered = {k: v for k, v in random_doc.items() if k in tagger_fields}
        # combine document field values into one string
        combined_texts = '\n'.join(random_doc_filtered.values())
        combined_texts, tags = self.get_mlp(combined_texts, lemmatize=False)
        # retrieve tag candidates
        tag_candidates = self.get_tag_candidates(combined_texts, ignore_tags=tags)
        # get tags
        tags += self.apply_tagger_group(random_doc_filtered, tag_candidates, request, input_type='doc')
        # return document with tags
        response = {"document": random_doc, "tags": tags}
        return Response(response, status=status.HTTP_200_OK)


    def apply_tagger_group(self, text, tag_candidates, request, input_type='text', lemmatize=False, feedback=False):
        # get tagger group object
        logging.getLogger(INFO_LOGGER).info(f"[Apply Tagger Group] Starting apply_tagger_group...")
        tagger_group_object = self.get_object()
        # get tagger objects
        candidates_str = "|".join(tag_candidates)
        tagger_objects = tagger_group_object.taggers.filter(description__iregex=f"^({candidates_str})$")
        # filter out completed
        tagger_objects = [tagger for tagger in tagger_objects if tagger.task.status == tagger.task.STATUS_COMPLETED]
        logging.getLogger(INFO_LOGGER).info(f"[Apply Tagger Group] Loaded {len(tagger_objects)} tagger objects.")
        # predict tags
        group_task = group(apply_tagger.s(tagger.pk, text, input_type=input_type, lemmatize=lemmatize, feedback=feedback) for tagger in tagger_objects)
        group_results = group_task.apply_async(queue=CELERY_SHORT_TERM_TASK_QUEUE)
        logging.getLogger(INFO_LOGGER).info(f"[Apply Tagger Group] Group task applied.")

        # retrieve results & remove non-hits
        tags = [tag for tag in group_results.get() if tag]
        logging.getLogger(INFO_LOGGER).info(f"[Apply Tagger Group] Retrieved results for {len(tags)} taggers.")
        # remove non-hits
        tags = [tag for tag in tags if tag['result']]
        logging.getLogger(INFO_LOGGER).info(f"[Apply Tagger Group] Retrieved {len(tags)} positive tags.")
        # if feedback was enabled, add urls
        tags = [add_finite_url_to_feedback(tag, request) for tag in tags]
        # sort by probability and return
        return sorted(tags, key=lambda k: k['probability'], reverse=True)
