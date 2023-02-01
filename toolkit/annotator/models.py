import json
from datetime import datetime
from typing import List, Optional

import elasticsearch_dsl
from django.contrib.auth.models import User
from django.db import models
from django.db.models import Q
from texta_elastic.core import ElasticCore
from texta_elastic.document import ESDocObject, ElasticDocument
from texta_elastic.searcher import ElasticSearcher

from toolkit.core.project.models import Project
from toolkit.core.task.models import Task
from toolkit.elastic.index.models import Index
from toolkit.model_constants import TaskModel
from toolkit.settings import CELERY_LONG_TERM_TASK_QUEUE, DESCRIPTION_CHAR_LIMIT, TEXTA_ANNOTATOR_KEY

# Create your models here.

# MySQL does not allow higher char limits for unique fields than 255.
# Although ES fields and others don't follow this convention, it's best to keep them short for now.
SIZE_LIMIT = 255

ANNOTATION_CHOICES = (
    ("binary", "binary"),
    ("multilabel", "multilabel"),
    ("entity", "entity")
)


class Labelset(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, default=None, null=True)
    indices = models.ManyToManyField(Index)
    fact_names = models.TextField(null=True)
    value_limit = models.IntegerField(null=True)
    category = models.CharField(max_length=50, default="")
    values = models.TextField(default=json.dumps([]))

    def __str__(self):
        return self.category


class MultilabelAnnotatorConfiguration(models.Model):
    labelset = models.ForeignKey(Labelset, on_delete=models.CASCADE)


class BinaryAnnotatorConfiguration(models.Model):
    fact_name = models.CharField(max_length=SIZE_LIMIT, help_text="Sets the value for the fact name for all annotated documents.")
    # Change these to a Label value.
    pos_value = models.CharField(max_length=SIZE_LIMIT, help_text="Sets the name for a fact value for positive documents.")
    neg_value = models.CharField(max_length=SIZE_LIMIT, help_text="Sets the name for a fact value for negative documents.")


class EntityAnnotatorConfiguration(models.Model):
    fact_name = models.CharField(max_length=SIZE_LIMIT, help_text="Name of the fact which will be added.")


class Annotator(TaskModel):
    annotator_uid = models.CharField(max_length=SIZE_LIMIT, default="", help_text="Unique Identifier for Annotator")
    annotation_type = models.CharField(max_length=SIZE_LIMIT, choices=ANNOTATION_CHOICES, help_text="Which type of annotation does the user wish to perform")

    annotator_users = models.ManyToManyField(User, default=None, related_name="annotators", help_text="Who are the users who will be annotating.")

    add_facts_mapping = models.BooleanField(default=True)
    use_shared_comments = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True, null=True)
    modified_at = models.DateTimeField(auto_now=True, null=True)
    completed_at = models.DateTimeField(null=True, default=None)

    total = models.IntegerField(default=0, help_text="How many documents are going to be annotated.")
    validated = models.IntegerField(default=0, help_text="How many documents of the total have been validated.")

    target_field = models.CharField(max_length=SIZE_LIMIT, default='', help_text="Which Elasticsearch document field you use base the annotation on.")

    binary_configuration = models.ForeignKey(
        BinaryAnnotatorConfiguration,
        on_delete=models.SET_NULL,
        null=True,
        default=None,
        help_text="Settings for binary type annotations."
    )
    multilabel_configuration = models.ForeignKey(
        MultilabelAnnotatorConfiguration,
        on_delete=models.SET_NULL,
        null=True,
        default=None,
        help_text="Settings for multilabel type annotations."
    )
    entity_configuration = models.ForeignKey(
        EntityAnnotatorConfiguration,
        on_delete=models.SET_NULL,
        null=True,
        default=None,
        help_text="Settings for entity type annotations."
    )

    @property
    def annotated(self):
        restraint = Record.objects.filter(annotated_utc__isnull=False, skipped_utc__isnull=True, annotation_job=self)
        return restraint.count()

    @property
    def skipped(self):
        restraint = Record.objects.filter(annotated_utc__isnull=True, skipped_utc__isnull=False, annotation_job=self)
        return restraint.count()

    def create_annotator_task(self):
        self.save()

        new_task = Task.objects.create(annotator=self, task_type=Task.TYPE_APPLY, status=Task.STATUS_CREATED)
        self.tasks.add(new_task)

        annotator_obj = Annotator.objects.get(pk=self.pk)
        annotator_group, is_created = AnnotatorGroup.objects.get_or_create(
            project=annotator_obj.project,
            parent=annotator_obj
        )

        from toolkit.annotator.tasks import annotator_task
        annotator_task.apply_async(args=(self.pk,), queue=CELERY_LONG_TERM_TASK_QUEUE)

    def add_pos_label(self, document_id: str, index: str, user):
        """
        Adds a positive label to the Elasticsearch document for Binary annotation.
        :param user: User that does the annotation.
        :param index: Which index does said Elasticsearch document reside in.
        :param document_id: Elasticsearch document ID of the comment in question.
        :return:
        """
        ed = ESDocObject(document_id=document_id, index=index)
        if "texta_facts" in ed.document["_source"]:
            for facts in ed.document["_source"]["texta_facts"]:
                if facts["fact"] == self.binary_configuration.fact_name and facts["source"] == "annotator":
                    if facts["str_val"] != self.binary_configuration.pos_value:
                        facts["str_val"] = self.binary_configuration.pos_value
                        ed.update()
                        return
                    else:
                        return
        fact = ed.add_fact(fact_value=self.binary_configuration.pos_value, fact_name=self.binary_configuration.fact_name, doc_path=self.target_field)
        ed.add_annotated(annotator_model=self, user=user)
        ed.update()
        self.generate_record(document_id, index=index, user_pk=user.pk, fact=fact, do_annotate=True, fact_id=fact["id"])

    def generate_record(self, document_id, index, user_pk, fact=None, fact_id=None, do_annotate=False, do_skip=False):
        user = User.objects.get(pk=user_pk)
        record, is_created = Record.objects.get_or_create(document_id=document_id, index=index, user=user, annotation_job=self)
        if do_annotate:
            record.skipped_utc = None
            record.fact = json.dumps(fact, ensure_ascii=False)
            record.fact_id = fact_id
            record.annotated_utc = datetime.utcnow()
        if do_skip:
            record.annotated_utc = None
            record.skipped_utc = datetime.utcnow()
        record.save()

    def add_neg_label(self, document_id: str, index: str, user):
        """
        Adds a negative label to the Elasticsearch document for Binary annotation.
        :param index: Which index does said Elasticsearch document reside in.
        :param document_id: Elasticsearch document ID of the comment in question.
        :return:
        """
        ed = ESDocObject(document_id=document_id, index=index)
        if "texta_facts" in ed.document["_source"]:
            for facts in ed.document["_source"]["texta_facts"]:
                if facts["fact"] == self.binary_configuration.fact_name and facts["source"] == "annotator":
                    if facts["str_val"] != self.binary_configuration.neg_value:
                        facts["str_val"] = self.binary_configuration.neg_value
                        ed.update()
                        return
                    else:
                        return
        fact = ed.add_fact(fact_value=self.binary_configuration.neg_value, fact_name=self.binary_configuration.fact_name, doc_path=self.target_field)
        ed.add_annotated(self, user)
        ed.update()
        self.generate_record(document_id, index=index, user_pk=user.pk, fact=fact, do_annotate=True, fact_id=fact["id"])

    def add_labels(self, document_id: str, labels: List[str], index: str, user: User):
        """
        Adds a label to Elasticsearch documents during multilabel annotations.
        :param index:
        :param document_id: Elasticsearch document ID of the comment in question.
        :param labels: Which labels to add into the document.
        :return:
        """
        ed = ESDocObject(document_id=document_id, index=index)
        if labels:
            for label in labels:
                fact = ed.add_fact(fact_value=label, fact_name=self.multilabel_configuration.labelset.category, doc_path=self.target_field, author=user.username)
                ed.add_annotated(self, user)
                self.generate_record(document_id, index=index, user_pk=user.pk, fact=fact, do_annotate=True, fact_id=fact["id"])
        else:
            ed.add_annotated(self, user)
            self.generate_record(document_id, index=index, user_pk=user.pk, fact=None, do_annotate=True, fact_id=None)

        ed.update()

    def __split_fact(self, fact: dict):
        fact_name, value, spans, field, fact_id = fact["fact"], fact.get("str_val") or fact.get("num_val"), fact.get("spans"), fact.get("doc_path"), fact.get("id", "")
        return fact_name, value, spans, field, fact_id

    def add_entity(self, document_id: str, texta_facts: List[dict], index: str, user: User):
        """
        Adds an entity label to Elasticsearch documents during entity annotations.
        :param user: Which user is adding the Facts.
        :param texta_facts: Facts to store inside Elasticsearch.
        :param index: Index into which the Facts are stored.
        :param document_id: Elasticsearch document ID of the comment in question.
        :return:
        """
        from toolkit.annotator.tasks import add_entity_task
        add_entity_task.apply_async(args=(self.pk, document_id, texta_facts, index, user.pk), queue=CELERY_LONG_TERM_TASK_QUEUE)

    def _generated_pull_by_counter_query(self, json_query: dict, document_counter: int) -> dict:
        import elasticsearch_dsl

        # Does not use the job id restriction as not all documents get it when before an annotation happens
        # and the singular index is enough to keep it restrained.
        positive_queries = [
            elasticsearch_dsl.Q(json_query["query"]),
            elasticsearch_dsl.Q("term", **{f"{TEXTA_ANNOTATOR_KEY}.document_counter": document_counter})
        ]
        search = elasticsearch_dsl.Search()
        restriction = elasticsearch_dsl.Q("bool", must=positive_queries)
        search = search.query(restriction)
        query = search.to_dict()
        return query

    def pull_document(self, document_counter: Optional[int]) -> Optional[dict]:
        """
        Function for returning a new Elasticsearch document for annotation.
        :return:
        """
        from texta_elastic.core import ElasticCore

        ec = ElasticCore()
        json_query = json.loads(self.query)
        indices = self.get_indices()

        if document_counter is None:
            query = ec.get_annotation_query(json_query, job_pk=self.pk)
        else:
            query = self._generated_pull_by_counter_query(json_query, document_counter)

        document = ESDocObject.random_document(indices=indices, query=query)
        # At one point in time, the documents will run out.
        if document:
            return document.document
        else:
            return None

    def skip_document(self, document_id: str, index: str, user) -> bool:
        """
        Add the skip label to the document and update the progress accordingly.
        :param user:
        :param index:
        :param document_id: Elasticsearch document ID of the comment in question.
        :return:
        """
        ed = ESDocObject(document_id=document_id, index=index)
        ed.add_skipped(self, user)
        ed.update()
        self.generate_record(document_id, index=index, user_pk=user.pk, do_skip=True)

        return True

    def __generate_annotator_meta(self, user: User) -> dict:
        """Function for facilitating a common structure for the annotator meta."""
        return {"job_id": self.pk, "user": user.username}

    def add_comment(self, document_id: str, comment: str, user: User) -> bool:
        """
        Adds an annotators comment into the document in question.
        :param user: Django user who did the comment.
        :param comment: Comment to be stores inside the Elasticsearch document.
        :param document_id: Elasticsearch document ID of the comment in question.
        :return:
        """
        indices = ",".join(self.get_available_or_all_indices())
        ed = ElasticDocument(index=indices)
        document = ed.get(document_id)
        source = document["_source"]
        document_uuid = source["texta_meta"]["document_uuid"]

        if TEXTA_ANNOTATOR_KEY not in source:
            source[TEXTA_ANNOTATOR_KEY] = {"comments": [comment], **self.__generate_annotator_meta(user)}
        else:
            comments = source[TEXTA_ANNOTATOR_KEY].get("comments", [])
            if comment not in comments:
                comments.append(comment)
            if comment not in source[TEXTA_ANNOTATOR_KEY]:
                source[TEXTA_ANNOTATOR_KEY]["comments"] = comments

        ed.update(index=document["_index"], doc_id=document_id, doc={TEXTA_ANNOTATOR_KEY: source[TEXTA_ANNOTATOR_KEY]})
        Comment.objects.create(annotation_job=self, text=comment, document_uuid=document_uuid, document_id=document_id, user=user)
        return True

    def get_comment_queryset(self, document_id: str, document_uuid: str, user: User):
        if self.use_shared_comments:
            queryset = Comment.objects.filter(
                Q(user__username=user.username, annotation_job__pk=self.pk, document_id=document_id) | Q(document_uuid=document_uuid)
            )
        else:
            queryset = Comment.objects.filter(user__username=user.username, annotation_job__pk=self.pk, document_id=document_id)
        return queryset

    def get_comments(self, document_id: str, user: User):
        """
        Adds an annotators comment into the document in question.
        :param user: Django user who is pulling the comments.
        :param document_id: Elasticsearch document ID of the comment in question.
        :return:
        """
        indices = ",".join(self.get_available_or_all_indices())
        ed = ElasticDocument(index=indices)
        document = ed.get(document_id)["_source"]
        document_uuid = document["texta_meta"]["document_uuid"]
        queryset = self.get_comment_queryset(document_id, document_uuid, user)
        return queryset.order_by("-created_at")

    def _get_skipped_document_query_with_counter(self, json_query, document_counter):
        negative_queries = [
            elasticsearch_dsl.Q("exists", field="texta_annotator.processed_timestamp_utc"),
            elasticsearch_dsl.Q("exists", field="texta_annotator.validated_timestamp_utc")
        ]
        positive_queries = [
            elasticsearch_dsl.Q(json_query["query"]),
            elasticsearch_dsl.Q("match", **{f"{TEXTA_ANNOTATOR_KEY}.job_id": self.pk}),
            elasticsearch_dsl.Q("exists", field=f"{TEXTA_ANNOTATOR_KEY}.skipped_timestamp_utc"),
            elasticsearch_dsl.Q("range", **{f"{TEXTA_ANNOTATOR_KEY}.document_counter": {"gte": document_counter}})
        ]
        s = elasticsearch_dsl.Q("bool", must_not=negative_queries, must=positive_queries)

        search = elasticsearch_dsl.Search().query(s)

        query = search.to_dict()
        return query

    def _pull_documents_for_counter_from_elasticsearch(self, ec: ElasticCore, query: dict, indices: List[str]):
        search = elasticsearch_dsl.Search \
            .from_dict(query) \
            .using(ec.es) \
            .index(",".join(indices)) \
            .sort({f"{TEXTA_ANNOTATOR_KEY}.document_counter": {"order": "asc"}})
        documents = list(search.execute())
        return documents

    def pull_skipped_document(self, document_counter: Optional[int]):
        """
        Returns all the documents that are marked for skipping.
        :return:
        """
        ec = ElasticCore()
        json_query = json.loads(self.query)
        indices = self.get_indices()

        if document_counter is None:
            query = ec.get_skipped_annotation_query(json_query, self.pk)
            document = ESDocObject.random_document(indices=indices, query=query)
            return document.document if document else None
        else:
            query = self._get_skipped_document_query_with_counter(json_query, document_counter)
            documents = self._pull_documents_for_counter_from_elasticsearch(ec, query, indices)
            document = documents[0] if documents else None
            return {"_index": document.meta.index, "_type": document.meta.doc_type, "_id": document.meta.id, "_source": document.to_dict()} if document else None

    def _get_annotated_document_query_with_counter(self, json_query, document_counter):
        positive_queries = [
            elasticsearch_dsl.Q(json_query["query"]),
            elasticsearch_dsl.Q("match", **{f"{TEXTA_ANNOTATOR_KEY}.job_id": self.pk}),
            elasticsearch_dsl.Q("exists", field=f"{TEXTA_ANNOTATOR_KEY}.processed_timestamp_utc"),
            elasticsearch_dsl.Q("range", **{f"{TEXTA_ANNOTATOR_KEY}.document_counter": {"gte": document_counter}})
        ]

        negative_queries = [
            elasticsearch_dsl.Q("exists", field=f"{TEXTA_ANNOTATOR_KEY}.skipped_timestamp_utc")
        ]

        s = elasticsearch_dsl.Q("bool", must_not=negative_queries, must=positive_queries)
        search = elasticsearch_dsl.Search().query(s)
        query = search.to_dict()
        return query

    def pull_annotated_document(self, document_counter: Optional[int]) -> Optional[dict]:
        """
        Returns an already annotated document for validation purposes.
        :return:
        """
        from texta_elastic.core import ElasticCore

        ec = ElasticCore()
        json_query = json.loads(self.query)
        indices = self.get_indices()

        if document_counter is None:
            query = ec.get_annotated_annotation_query(query=json_query, job_pk=self.pk)
            document = ESDocObject.random_document(indices=indices, query=query)
            return document.document if document else None
        else:
            query = self._get_annotated_document_query_with_counter(json_query, document_counter)
            documents = self._pull_documents_for_counter_from_elasticsearch(ec, query, indices)
            document = documents[0] if documents else None
            return {"_index": document.meta.index, "_type": document.meta.doc_type, "_id": document.meta.id, "_source": document.to_dict()} if document else None

    def pull_commented_document(self, user: User, document_counter: Optional[int]) -> Optional[dict]:
        """
        Returns an already annotated document for validation purposes.
        :return:
        """
        from texta_elastic.core import ElasticCore

        ec = ElasticCore()
        json_query = json.loads(self.query)
        indices = self.get_indices()

        # Not all commented documents are annotated and thus get the job_id and user values, hence we only use query and the existence of comments.
        base_positives = [
            elasticsearch_dsl.Q(json_query["query"]),
            elasticsearch_dsl.Q("exists", field=f"{TEXTA_ANNOTATOR_KEY}.comments")
        ]

        if document_counter is None:
            s = elasticsearch_dsl.Q("bool", must=base_positives, must_not=elasticsearch_dsl.Q("terms", **{f"{TEXTA_ANNOTATOR_KEY}.comments": []}))
            query = s.to_dict()
            document = ESDocObject.random_document(indices=indices, query=query)
            return document.document if document else None

        else:
            base_positives.append(elasticsearch_dsl.Q("range", **{f"{TEXTA_ANNOTATOR_KEY}.document_counter": {"gte": document_counter}}))
            s = elasticsearch_dsl.Q("bool", must=base_positives, must_not=elasticsearch_dsl.Q("terms", **{f"{TEXTA_ANNOTATOR_KEY}.comments": []}))
            search = elasticsearch_dsl.Search().query(s)
            query = search.to_dict()
            documents = self._pull_documents_for_counter_from_elasticsearch(ec, query, indices)
            document = documents[0] if documents else None
            return {"_index": document.meta.index, "_type": document.meta.doc_type, "_id": document.meta.id, "_source": document.to_dict()} if document else None

    def reset_processed_records(self, indices: List[str], query: dict):
        """
        Resets the timestamp for all documents matching the query that have the "processed" timestamp.
        :param indices: Which indices to target for the rest.
        :param query: Elasticsearch query to subset the documents of the indices for the reset.
        :return:
        """
        pass

    @staticmethod
    def add_texta_meta_mapping(indices: List[str]):
        """
        Adds the mapping for texta_meta.
        :param indices: Which indices to target for the schemas.
        :return:
        """
        from texta_elastic.core import ElasticCore

        ec = ElasticCore()
        for index in indices:
            ec.add_texta_meta_mapping(index)


class AnnotatorGroup(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, default=None, null=True)
    parent = models.ForeignKey(Annotator, on_delete=models.CASCADE)
    children = models.ManyToManyField(Annotator, default=None, related_name="annotator_group_children")


class Comment(models.Model):
    text = models.TextField()
    document_id = models.CharField(max_length=DESCRIPTION_CHAR_LIMIT)
    document_uuid = models.CharField(max_length=DESCRIPTION_CHAR_LIMIT, default="")
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    annotation_job = models.ForeignKey(Annotator, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return f"{self.user.username}: {self.text} @{str(self.created_at)}"


class Record(models.Model):
    document_id = models.CharField(max_length=SIZE_LIMIT, db_index=True, help_text="Elasticsearch document ID for the sub-index.")
    document_uuid = models.CharField(max_length=SIZE_LIMIT, default=None, null=True, help_text="Reference UUID for the original document.")
    index = models.CharField(max_length=SIZE_LIMIT, help_text="Which Elasticsearch index does the document live in.")
    fact_id = models.TextField(default=None, null=True, db_index=True, help_text="UUID of the Texta Fact, useful for editing later.")

    fact = models.TextField(default=json.dumps({}))

    annotated_utc = models.DateTimeField(default=None, null=True)
    skipped_utc = models.DateTimeField(default=None, null=True)

    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    annotation_job = models.ForeignKey(Annotator, on_delete=models.SET_NULL, null=True)
