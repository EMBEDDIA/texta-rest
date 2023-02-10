# Create your tests here.
import json
from typing import Optional, List

from django.contrib.auth.models import User
from django.test import override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from texta_elastic.core import ElasticCore

from toolkit.annotator.models import Annotator, Labelset
from toolkit.elastic.index.models import Index
from toolkit.helper_functions import reindex_test_dataset
from toolkit.settings import TEXTA_ANNOTATOR_KEY, TEXTA_TAGS_KEY
from toolkit.test_settings import TEST_FIELD, TEST_MATCH_TEXT, TEST_QUERY
from toolkit.tools.utils_for_tests import create_test_user, print_output, project_creation


class TestHelpers(APITestCase):

    # Check for response status is kept out since in certain cases a 404 is what we want.
    def skip_document(self, document_id: str, index: str, project_pk, annotator_pk):
        url = reverse("v2:annotator-skip-document", kwargs={"project_pk": project_pk, "pk": annotator_pk})
        response = self.client.post(url, data={"document_id": document_id, "index": index}, format="json")
        print_output("skip_document:response.data", response.data)
        return response

    def pull_random_document(self, project_pk: int, annotator_pk: int, document_counter: Optional[int] = None):
        url = reverse("v2:annotator-pull-document", kwargs={"project_pk": project_pk, "pk": annotator_pk})
        kwargs = {"document_counter": document_counter} if document_counter is not None else {}
        response = self.client.post(url, data=kwargs, format="json")
        return response

    def pull_skipped_document(self, project_pk: int, annotator_pk: int, document_counter: Optional[int] = None):
        url = reverse("v2:annotator-pull-skipped", kwargs={"project_pk": project_pk, "pk": annotator_pk})
        kwargs = {"document_counter": document_counter} if document_counter is not None else {}
        response = self.client.post(url, data=kwargs, format="json")
        return response

    def check_fact_structure(self, index: str, document_id: str, str_vals: List[str], user: User):
        ec = ElasticCore()
        document = ec.es.get(index=index, id=document_id)
        facts = document["_source"][TEXTA_TAGS_KEY]
        self.assertTrue(len(facts) > 0)
        fact = facts[0]
        self.assertEqual(fact["source"], "annotator")
        self.assertTrue(fact["id"])
        self.assertEqual(fact["author"], user.username)
        self.assertTrue(fact["str_val"] in str_vals)


@override_settings(CELERY_ALWAYS_EAGER=True)
class BinaryAnnotatorTests(TestHelpers):

    def setUp(self):
        # Owner of the project
        self.test_index_name = reindex_test_dataset(limit=10)
        self.secondary_index = reindex_test_dataset(limit=10)
        self.index, is_created = Index.objects.get_or_create(name=self.secondary_index)
        self.user = create_test_user('annotator', 'my@email.com', 'pw')
        self.user2 = create_test_user('annotator2', 'test@email.com', 'pw2')
        self.project = project_creation("taggerTestProject", self.test_index_name, self.user)
        self.project.indices.add(self.index)
        self.project.users.add(self.user)
        self.project.users.add(self.user2)

        self.client.login(username='annotator', password='pw')
        self.ec = ElasticCore()

        self.list_view_url = reverse("v2:annotator-list", kwargs={"project_pk": self.project.pk})
        self.annotator = self._create_annotator()
        self.pull_document_url = reverse("v2:annotator-pull-document", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})

    def test_all(self):
        self.run_binary_annotator_group()
        self.run_create_annotator_for_multi_user()
        self.run_pulling_document()
        self.run_binary_annotation()
        # self.run_that_query_limits_pulled_document()
        doc_id_with_comment = self.run_adding_comment_to_document()
        self.run_pulling_comment_for_document(doc_id_with_comment)
        self.run_check_proper_skipping_functionality()
        # self.run_annotating_to_the_end()

    def _create_annotator(self):
        payload = {
            "description": "Random test annotation.",
            "indices": [{"name": self.test_index_name}, {"name": self.secondary_index}],
            "fields": ["comment_content", TEST_FIELD],
            "target_field": "comment_content",
            "annotation_type": "binary",
            "annotating_users": ["annotator"],
            "binary_configuration": {
                "fact_name": "TOXICITY",
                "pos_value": "DO_DELETE",
                "neg_value": "SAFE"
            }
        }
        response = self.client.post(self.list_view_url, data=payload, format="json")
        print_output("_create_annotator:response.status", response.status_code)
        self.assertTrue(response.status_code == status.HTTP_201_CREATED)

        total_count = self.ec.es.count(index=f"{self.test_index_name},{self.secondary_index}").get("count", 0)
        self.assertTrue(total_count >= response.data["total"])
        return response.data

    def run_binary_annotator_group(self):
        annotator_children = []
        for i in range(2):
            child = self._create_annotator()
            annotator_children.append(child["id"])
        group_url = reverse("v2:annotator_groups-list", kwargs={"project_pk": self.project.pk})
        group_payload = {
            "parent": self.annotator["id"],
            "children": annotator_children
        }
        group_response = self.client.post(group_url, data=group_payload, format="json")
        print_output("run_binary_annotator_group:response.status", group_response.status_code)
        self.assertTrue(group_response.status_code == status.HTTP_201_CREATED)

    def run_create_annotator_for_multi_user(self):
        payload = {
            "description": "Multi user annotation.",
            "indices": [{"name": self.test_index_name}, {"name": self.secondary_index}],
            "fields": ["comment_content", TEST_FIELD],
            "target_field": "comment_content",
            "annotation_type": "binary",
            "annotating_users": ["annotator", "annotator2"],
            "binary_configuration": {
                "fact_name": "TOXICITY",
                "pos_value": "DO_DELETE",
                "neg_value": "SAFE"
            }
        }
        response = self.client.post(self.list_view_url, data=payload, format="json")
        print_output("create_annotator_for_multi_user:response.status", response.status_code)
        print_output("create_annotator_for_multi_user:response.data", response.data)
        self.assertTrue(response.status_code == status.HTTP_201_CREATED)
        for d in response.data["annotator_users"]:
            self.assertIn(d["username"], {str(self.user), str(self.user2)})

        total_count = self.ec.es.count(index=f"{self.test_index_name},{self.secondary_index}").get("count", 0)
        self.assertTrue(total_count >= response.data["total"])

    def run_binary_annotation(self):
        annotation_url = reverse("v2:annotator-annotate-binary", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})
        print_output("run_binary_annotation:annotation_url", annotation_url)
        annotation_payloads = []
        for i in range(2):
            random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
            es_index = random_document["_index"]
            es_doc_id = random_document["_id"]
            annotation_payloads.append(
                {"index": es_index, "document_id": es_doc_id, "doc_type": "_doc", "annotation_type": "pos"}
            )
        print_output("annotation_document_before_0", annotation_payloads[0]['document_id'])
        print_output("annotation_document_before_1", annotation_payloads[1]['document_id'])
        while annotation_payloads[0]['document_id'] == annotation_payloads[1]['document_id']:
            random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
            annotation_payloads[1] = {"index": random_document["_index"], "document_id": random_document["_id"], "doc_type": "_doc", "annotation_type": "pos"}
        print_output("run_binary_annotation:annotation_payloads", annotation_payloads)
        for index_count, payload in enumerate(annotation_payloads):
            print_output(f"run_binary_annotation:annotation_payload{index_count}", payload['document_id'])
            annotation_response = self.client.post(annotation_url, data=payload, format="json")
            # Test for response success.
            print_output("run_binary_annotation:response.status", annotation_response.status_code)
            self.assertTrue(annotation_response.status_code == status.HTTP_200_OK)

            # Test that progress is updated properly.
            model_object = Annotator.objects.get(pk=self.annotator["id"])
            print_output("run_binary_annotation:annotator_model_obj", model_object.annotated)
            print_output("run_binary_annotation:binary_index_count", index_count)
            self.assertTrue(model_object.annotated == index_count + 1)

            # Check that document was actually edited.
            es_index = payload["index"]
            es_doc_id = payload["document_id"]
            es_doc = self.ec.es.get(index=es_index, id=es_doc_id)["_source"]
            facts = es_doc["texta_facts"]
            self.assertTrue(model_object.binary_configuration.fact_name in [fact["fact"] for fact in facts])
            if payload["annotation_type"] == "pos":
                self.assertTrue(model_object.binary_configuration.pos_value in [fact["str_val"] for fact in facts])
            elif payload["annotation_type"] == "neg":
                self.assertTrue(model_object.binary_configuration.neg_value in [fact["str_val"] for fact in facts])

            self.check_fact_structure(es_index, es_doc_id, [model_object.binary_configuration.neg_value, model_object.binary_configuration.pos_value], self.user)

    # def run_annotating_to_the_end(self):
    #     model_object = Annotator.objects.get(pk=self.annotator["id"])
    #     total = model_object.total
    #     annotated = model_object.annotated
    #     skipped = model_object.skipped
    #
    #     annotation_url = reverse("v2:annotator-annotate-binary", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})
    #
    #     for i in range(1, total - annotated - skipped):
    #         random_document = self._pull_random_document()
    #         payload = {"annotation_type": "pos", "document_id": random_document["_id"], "index": random_document["_index"]}
    #         annotation_response = self.client.post(annotation_url, data=payload, format="json")
    #         self.assertTrue(annotation_response.status_code == status.HTTP_200_OK)
    #
    #     # At this point all the documents should be done.
    #     random_document = self._pull_random_document()
    #     self.assertTrue(random_document["detail"] == 'No more documents left!')

    def run_pulling_comment_for_document(self, document_id):
        url = reverse("v2:annotator-get-comments", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})
        payload = {"document_id": document_id}
        response = self.client.post(url, data=payload, format="json")
        print_output("run_pulling_comment_for_document:response.data", response.status_code)
        self.assertTrue(response.status_code == status.HTTP_200_OK)
        self.assertTrue(response.data["count"] == 1)

        comment = response.data["results"][0]
        self.assertTrue(comment.get("text", ""))
        self.assertTrue(comment.get("document_id", "") == document_id)
        self.assertTrue(comment.get("user", "") == self.user.username)
        self.assertTrue(comment.get("created_at", ""))

    def run_pulling_document(self):
        url = reverse("v2:annotator-pull-document", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})
        # Test pulling documents several times as that will be a common behavior.
        for i in range(3):
            response = self.client.post(url, format="json")
            print_output("run_pulling_document:response.status", response.status_code)
            self.assertTrue(response.status_code == status.HTTP_200_OK)
            self.assertTrue(response.data.get("_id", None))
            self.assertTrue(response.data.get("_source", None))
            self.assertTrue(response.data.get("_index", None))
            self.assertTrue(response.data["_source"].get(TEST_FIELD), None)

    def run_adding_comment_to_document(self):
        random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
        document_id = random_document["_id"]
        url = reverse("v2:annotator-add-comment", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})
        payload = {
            "document_id": document_id,
            "text": "Ah, miks sa teed nii!?"
        }
        response = self.client.post(url, data=payload, format="json")
        print_output("run_adding_comment_to_document:response.status", response.status_code)
        self.assertTrue(response.status_code == status.HTTP_200_OK)
        return document_id

    def run_check_proper_skipping_functionality(self):
        random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
        skip_url = reverse("v2:annotator-skip-document", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})
        doc_index = random_document["_index"]
        doc_id = random_document["_id"]
        response = self.client.post(skip_url, data={"document_id": doc_id, "index": doc_index}, format="json")
        print_output("run_check_proper_skipping_functionality:response.status", response.status_code)
        self.assertTrue(response.status_code == status.HTTP_200_OK)
        elastic_doc = self.ec.es.get(index=doc_index, id=doc_id)

        annotator_dict = elastic_doc["_source"][TEXTA_ANNOTATOR_KEY]
        self.assertTrue(annotator_dict["skipped_timestamp_utc"])

    def test_that_double_skipped_document_wont_be_counted(self):
        random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
        skip = self.skip_document(random_document["_id"], random_document["_index"], self.project.pk, self.annotator["id"])
        first_count = Annotator.objects.get(pk=self.annotator["id"]).skipped
        skip = self.skip_document(random_document["_id"], random_document["_index"], self.project.pk, self.annotator["id"])
        second_count = Annotator.objects.get(pk=self.annotator["id"]).skipped
        self.assertEqual(first_count, second_count)

    def test_overwrite_of_meta_fields_after_annotating_skipped_document_and_proper_structure_of_skipped_documents(self):
        ec = ElasticCore()
        random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
        es_id = random_document["_id"]
        es_index = random_document["_index"]

        # We do this since in some cases you can have other pre-annotation facts in Annotator.
        facts = [fact["str_val"] for fact in random_document["_source"].get(TEXTA_TAGS_KEY, []) if fact.get("str_val", None)]
        self.assertTrue(self.annotator["binary_configuration"]["pos_value"] not in facts)
        self.assertTrue(self.annotator["binary_configuration"]["neg_value"] not in facts)

        # Check that counts are updated properly.
        initial_skipped_count = Annotator.objects.get(pk=self.annotator["id"]).skipped
        initial_annotated_count = Annotator.objects.get(pk=self.annotator["id"]).annotated
        skip = self.skip_document(es_id, es_index, self.project.pk, self.annotator["id"])
        post_skip_count = Annotator.objects.get(pk=self.annotator["id"]).skipped
        self.assertEqual(initial_skipped_count + 1, post_skip_count)
        self.assertEqual(initial_annotated_count, 0)

        # Make the assertions for a proper skip.
        document = ec.es.get(index=es_index, id=es_id)
        self.assertTrue(TEXTA_ANNOTATOR_KEY in document["_source"])
        self.assertTrue("skipped_timestamp_utc" in document["_source"][TEXTA_ANNOTATOR_KEY])
        self.assertTrue("processed_timestamp_utc" not in document["_source"][TEXTA_ANNOTATOR_KEY])
        # Since it's the first document we skipped in this test it should return the same document.
        skipped_document = self.pull_skipped_document(self.project.pk, self.annotator["id"])
        self.assertEqual(skipped_document.status_code, status.HTTP_200_OK)
        self.assertEqual(skipped_document.data["_id"], es_id)

        # Annotate the skipped document.
        annotation_url = reverse("v2:annotator-annotate-binary", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})
        annotation_response = self.client.post(
            annotation_url,
            data={"annotation_type": "pos", "document_id": random_document["_id"], "index": random_document["_index"]},
            format="json"
        )
        self.assertEqual(annotation_response.status_code, status.HTTP_200_OK)

        # Check that the annotation count is increased and the skipped one decreased back to initial.
        post_annotation_skipped_count = Annotator.objects.get(pk=self.annotator["id"]).skipped
        post_annotation_annotated_count = Annotator.objects.get(pk=self.annotator["id"]).annotated
        self.assertEqual(initial_skipped_count, post_annotation_skipped_count)
        self.assertEqual(initial_annotated_count + 1, post_annotation_annotated_count)

        skipped_document = self.pull_skipped_document(self.project.pk, self.annotator["id"])
        self.assertEqual(skipped_document.status_code, status.HTTP_404_NOT_FOUND)

        # Ensure the Elasticsearch changes are what are expected.
        document = ec.es.get(index=es_index, id=es_id)
        self.assertTrue(TEXTA_ANNOTATOR_KEY in document["_source"])
        self.assertTrue("skipped_timestamp_utc" not in document["_source"][TEXTA_ANNOTATOR_KEY])
        self.assertTrue("processed_timestamp_utc" in document["_source"][TEXTA_ANNOTATOR_KEY])
        self.assertTrue(document["_source"].get(TEXTA_TAGS_KEY, []) != [])

    def test_that_going_through_skipped_documents_is_rotating(self):
        random_document = self.pull_random_document(self.project.pk, self.annotator["id"], document_counter=1).data
        skip = self.skip_document(random_document["_id"], random_document["_index"], self.project.pk, self.annotator["id"])
        second_random = self.pull_random_document(self.project.pk, self.annotator["id"], document_counter=5).data
        second_skip = self.skip_document(second_random["_id"], second_random["_index"], self.project.pk, self.annotator["id"])

        # Since there is no skipped document with a counter of zero it should upgrade and pull the one with 1.
        first_skipped = self.pull_skipped_document(self.project.pk, self.annotator["id"], document_counter=0).data
        self.assertTrue(first_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] == 1)

        # Check that it fetches the next one since there is no skipped document with 2.
        next_counter = first_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] + 1
        second_skipped = self.pull_skipped_document(self.project.pk, self.annotator["id"], document_counter=next_counter).data
        self.assertTrue(second_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] == 5)

        # Since there are only two skipped documents, trying to go ahead of the latest one it should rotate back.
        next_counter = second_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] + 1
        third_skipped = self.pull_skipped_document(self.project.pk, self.annotator["id"], document_counter=next_counter).data
        self.assertTrue(third_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] == 1)

    # Test for an old bug where every single Annotator instance was counted in the counts.
    def test_that_annotator_count_counts_parent_tasks_only(self):
        annotator = self._create_annotator()
        uri = reverse("v2:project-get-resource-counts", kwargs={"pk": self.project.pk})
        response = self.client.get(uri)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["num_annotators"], 2)


@override_settings(CELERY_ALWAYS_EAGER=True)
class EntityAnnotatorTests(TestHelpers):

    def setUp(self):
        # Owner of the project
        self.test_index_name = reindex_test_dataset(limit=10)
        self.secondary_index = reindex_test_dataset(limit=10)
        self.index, is_created = Index.objects.get_or_create(name=self.secondary_index)
        self.user = create_test_user('annotator', 'my@email.com', 'pw')
        self.user2 = create_test_user('annotator2', 'test@email.com', 'pw2')
        self.project = project_creation("entityTestProject", self.test_index_name, self.user)
        self.project.indices.add(self.index)
        self.project.users.add(self.user)
        self.project.users.add(self.user2)

        self.client.login(username='annotator', password='pw')
        self.ec = ElasticCore()

        self.list_view_url = reverse("v2:annotator-list", kwargs={"project_pk": self.project.pk})
        self.annotator = self._create_annotator()
        self.pull_document_url = reverse("v2:annotator-pull-document", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})

    def test_all(self):
        self.run_entity_annotator_group()
        self.run_entity_annotation()

    def _create_annotator(self):
        payload = {
            "description": "Random test annotation.",
            "indices": [{"name": self.test_index_name}, {"name": self.secondary_index}],
            "fields": [TEST_FIELD],
            "annotating_users": ["annotator"],
            "annotation_type": "entity",
            "entity_configuration": {
                "fact_name": "TOXICITY"
            }
        }
        response = self.client.post(self.list_view_url, data=payload, format="json")
        print_output("_create_annotator:response.data", response.data)
        self.assertTrue(response.status_code == status.HTTP_201_CREATED)

        total_count = self.ec.es.count(index=f"{self.test_index_name},{self.secondary_index}").get("count", 0)
        self.assertTrue(total_count >= response.data["total"])
        return response.data

    def run_entity_annotator_group(self):
        annotator_children = []
        for i in range(2):
            child = self._create_annotator()
            annotator_children.append(child["id"])
        group_url = reverse("v2:annotator_groups-list", kwargs={"project_pk": self.project.pk})
        group_payload = {
            "parent": self.annotator["id"],
            "children": annotator_children
        }
        group_response = self.client.post(group_url, data=group_payload, format="json")
        print_output("run_entity_annotator_group:response.status", group_response.status_code)
        self.assertTrue(group_response.status_code == status.HTTP_201_CREATED)

    def run_entity_annotation(self):
        annotation_url = reverse("v2:annotator-annotate-entity", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})
        print_output("run_entity_annotation:annotation_url", annotation_url)
        annotation_payloads = []
        for i in range(2):
            random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
            annotation_payloads.append(
                {
                    "index": random_document["_index"], "document_id": random_document["_id"],
                    "texta_facts": [{"doc_path": TEST_FIELD, "fact": "TOXICITY", "spans": "[[0,0]]", "str_val": "bar", "source": "annotator"}]
                }
            )
        print_output("annotation_document_before_0", annotation_payloads[0]['document_id'])
        print_output("annotation_document_before_1", annotation_payloads[1]['document_id'])
        while annotation_payloads[0]['document_id'] == annotation_payloads[1]['document_id']:
            random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
            annotation_payloads[1] = {
                "index": random_document["_index"], "document_id": random_document["_id"],
                "texta_facts": [{"doc_path": TEST_FIELD, "fact": "TOXICITY", "spans": "[[0,0]]", "str_val": "bar", "source": "annotator"}]
            }
        print_output("run_entity_annotation:annotation_payloads", annotation_payloads)
        for index_count, payload in enumerate(annotation_payloads):
            print_output(f"run_entity_annotation:annotation_payload{index_count}", payload['document_id'])
            annotation_response = self.client.post(annotation_url, data=payload, format="json")
            # Test for response success.
            print_output("run_entity_annotation:response.status", annotation_response.status_code)
            self.assertTrue(annotation_response.status_code == status.HTTP_200_OK)

            # Test that progress is updated properly.
            model_object = Annotator.objects.get(pk=self.annotator["id"])
            print_output("run_entity_annotation:annotator_model_obj", model_object.annotated)
            print_output("run_entity_annotation:entity_index_count", index_count)
            self.assertTrue(model_object.annotated == index_count + 1)

            # Check that document was actually edited.
            es_index = payload["index"]
            es_doc_id = payload["document_id"]
            es_doc = self.ec.es.get(index=es_index, id=es_doc_id)["_source"]
            facts = es_doc["texta_facts"]
            print_output("facts", facts)
            print_output("mode_object", model_object.entity_configuration.fact_name)
            self.assertTrue(model_object.entity_configuration.fact_name in [fact["fact"] for fact in facts])

            self.check_fact_structure(es_index, es_doc_id, ["bar"], self.user)

    def test_that_double_skipped_document_wont_be_counted(self):
        random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
        skip = self.skip_document(random_document["_id"], random_document["_index"], self.project.pk, self.annotator["id"])
        first_count = Annotator.objects.get(pk=self.annotator["id"]).skipped
        skip = self.skip_document(random_document["_id"], random_document["_index"], self.project.pk, self.annotator["id"])
        second_count = Annotator.objects.get(pk=self.annotator["id"]).skipped
        self.assertEqual(first_count, second_count)

    def test_that_going_through_skipped_documents_is_rotating(self):
        random_document = self.pull_random_document(self.project.pk, self.annotator["id"], document_counter=1).data
        skip = self.skip_document(random_document["_id"], random_document["_index"], self.project.pk, self.annotator["id"])
        second_random = self.pull_random_document(self.project.pk, self.annotator["id"], document_counter=5).data
        second_skip = self.skip_document(second_random["_id"], second_random["_index"], self.project.pk, self.annotator["id"])

        # Since there is no skipped document with a counter of zero it should upgrade and pull the one with 1.
        first_skipped = self.pull_skipped_document(self.project.pk, self.annotator["id"], document_counter=0).data
        self.assertTrue(first_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] == 1)

        # Check that it fetches the next one since there is no skipped document with 2.
        next_counter = first_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] + 1
        second_skipped = self.pull_skipped_document(self.project.pk, self.annotator["id"], document_counter=next_counter).data
        self.assertTrue(second_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] == 5)

        # Since there are only two skipped documents, trying to go ahead of the latest one it should rotate back.
        next_counter = second_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] + 1
        third_skipped = self.pull_skipped_document(self.project.pk, self.annotator["id"], document_counter=next_counter).data
        self.assertTrue(third_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] == 1)


@override_settings(CELERY_ALWAYS_EAGER=True)
class MultilabelAnnotatorTests(TestHelpers):

    def setUp(self):
        # Owner of the project
        self.test_index_name = reindex_test_dataset(limit=10)
        self.index, is_created = Index.objects.get_or_create(name=self.test_index_name)
        self.user = create_test_user('annotator', 'my@email.com', 'pw')
        self.project = project_creation("multilabelTestProject", self.test_index_name, self.user)
        self.project.indices.add(self.index)
        self.project.users.add(self.user)

        self.client.login(username='annotator', password='pw')
        self.ec = ElasticCore()

        self.list_view_url = reverse("v2:annotator-list", kwargs={"project_pk": self.project.pk})
        self.labelset_url = reverse("v2:labelset-list", kwargs={"project_pk": self.project.pk})
        self.get_facts_url = reverse("v2:get_facts", kwargs={"project_pk": self.project.pk})
        self.facts = self._get_facts()
        self.labelset = self._create_labelset()
        self.annotator = self._create_annotator()
        self.pull_document_url = reverse("v2:annotator-pull-document", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})

    def test_all(self):
        self.run_multilabel_annotation()
        self.run_empty_multilabel_annotation()
        self.run_test_that_you_can_edit_labels_and_category()

    def _create_annotator(self):
        payload = {
            "description": "Random test annotation.",
            "indices": [{"name": self.test_index_name}],
            "fields": [TEST_FIELD],
            "annotating_users": ["annotator"],
            "annotation_type": "multilabel",
            "multilabel_configuration": {
                "labelset": self.labelset["id"]
            }
        }
        response = self.client.post(self.list_view_url, data=payload, format="json")
        print_output("_create_annotator:response.status", response.status_code)
        self.assertTrue(response.status_code == status.HTTP_201_CREATED)
        return response.data

    def _get_facts(self):
        payload = {}
        response = self.client.post(self.get_facts_url, data=payload, format="json")
        print_output("_get_facts:response.status", response.status_code)
        self.assertTrue(response.status_code == status.HTTP_200_OK)
        return response.data

    def _get_comments(self, document_id: str, annotator_pk: int):
        url = reverse("v2:annotator-get-comments", kwargs={"project_pk": self.project.pk, "pk": annotator_pk})
        response = self.client.post(url, data={"document_id": document_id}, format="json")
        print_output("_get_comments:response.data", response.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        return response.data

    def _create_labelset(self):
        payload = {
            "category": "new",
            "value_limit": 500,
            "indices": [self.test_index_name],
            "fact_names": [self.facts[0]["name"]],
            "values": ["true"]
        }
        response = self.client.post(self.labelset_url, data=payload, format="json")
        print_output("_create_labelset:response.status", response.status_code)
        self.assertTrue(response.status_code == status.HTTP_201_CREATED)

        self.assertTrue("true" in response.data["values"])
        self.assertTrue(Labelset.objects.count() != 0)
        return response.data

    def _add_comment_to_document(self, document_id: str, text: str, annotator_pk: int):
        url = reverse("v2:annotator-add-comment", kwargs={"project_pk": self.project.pk, "pk": annotator_pk})
        response = self.client.post(url, data={"document_id": document_id, "text": text}, format="json")
        print_output("_add_comment_to_document:response.data", response.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def run_multilabel_annotation(self):
        annotation_url = reverse("v2:annotator-annotate-multilabel", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})
        print_output("run_multilabel_annotation:annotation_url", annotation_url)
        random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data

        # Get model object before annotation.
        model_object_before = Annotator.objects.get(pk=self.annotator["id"])
        annotations_before = model_object_before.annotated
        print_output("run_multilabel_annotation:annotations_before", annotations_before)

        es_index = random_document["_index"]
        es_doc_id = random_document["_id"]
        annotation_payload = {
            "document_id": es_doc_id,
            "index": es_index,
            "labels": self.labelset["values"]
        }

        print_output(f"run_multilabel_annotation:annotation_payload", annotation_payload['document_id'])
        annotation_response = self.client.post(annotation_url, data=annotation_payload, format="json")
        # Test for response success.
        print_output("run_multilabel_annotation:response.status", annotation_response.status_code)
        self.assertTrue(annotation_response.status_code == status.HTTP_200_OK)

        # Test that progress is updated properly.
        model_object_after = Annotator.objects.get(pk=self.annotator["id"])
        annotations_after = model_object_after.annotated
        print_output("run_multilabel_annotation:annotations_after", annotations_after)
        self.assertTrue(annotations_after == annotations_before + 1)

        self.check_fact_structure(es_index, es_doc_id, self.labelset["values"], self.user)

    def run_empty_multilabel_annotation(self):
        annotation_url = reverse("v2:annotator-annotate-multilabel", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})
        print_output("run_multilabel_annotation:annotation_url", annotation_url)
        random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data

        # Get model object before annotation.
        model_object_before = Annotator.objects.get(pk=self.annotator["id"])
        annotations_before = model_object_before.annotated
        print_output("run_empty_multilabel_annotation:annotations_before", annotations_before)

        annotation_payload = {
            "document_id": random_document["_id"],
            "index": random_document["_index"],
            "labels": []
        }

        print_output(f"run_empty_multilabel_annotation:annotation_payload", annotation_payload['document_id'])
        annotation_response = self.client.post(annotation_url, data=annotation_payload, format="json")
        # Test for response success.
        print_output("run_empty_multilabel_annotation:response.status", annotation_response.status_code)
        self.assertTrue(annotation_response.status_code == status.HTTP_200_OK)

        # Test that progress is updated properly.
        model_object_after = Annotator.objects.get(pk=self.annotator["id"])
        annotations_after = model_object_after.annotated
        print_output("run_empty_multilabel_annotation:annotations_after", annotations_after)
        self.assertTrue(annotations_after == annotations_before + 1)

    def run_test_that_you_can_edit_labels_and_category(self):
        url = reverse("v2:labelset-detail", kwargs={"project_pk": self.project.pk, "pk": self.labelset["id"]})
        detail_response = self.client.get(url)
        is_list = isinstance(detail_response.data["values"], list)
        self.assertTrue(is_list)

        new_category = "booleans galore"
        new_values = ["true", "false", "1", "0"]
        payload = {
            "values": new_values,
            "category": new_category
        }
        edit_response = self.client.patch(url, data=payload, format="json")
        print_output("run_test_that_you_can_edit_labels_and_category:response.data", edit_response.data)
        self.assertEqual(edit_response.status_code, status.HTTP_200_OK)
        detail_response = self.client.get(url)
        self.assertTrue(detail_response.data["values"] == new_values)
        self.assertTrue(detail_response.data["category"] == new_category)

    def _get_second_unique_document(self, first_id: str, max_limit=100):
        counter = 0
        document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
        while document["_id"] == first_id and counter <= max_limit:
            document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
            counter += 1

        if counter == max_limit - 1:
            raise ValueError("Didn't find unique document!")

        return document

    # Test case for a bug where the user would see all of their comments through every document.
    def test_that_comments_are_visible_only_from_their_specific_document(self):
        first_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
        first_id = first_document["_id"]
        first_text = "Hello there, kenobi"

        second_document = self._get_second_unique_document(first_id)
        second_id = second_document["_id"]
        second_text = "The number of seconds you shall wait is 3, not 2, 4 is totally out of question."

        self._add_comment_to_document(first_id, first_text, self.annotator["id"])
        self._add_comment_to_document(second_id, second_text, self.annotator["id"])

        first_comments = self._get_comments(first_id, self.annotator["id"])
        first_comments = first_comments["results"]
        self.assertEqual(len(first_comments), 1)
        self.assertTrue(first_comments[0]["text"] == first_text)
        self.assertTrue(first_comments[0]["document_id"] == first_id)

        second_comments = self._get_comments(second_id, self.annotator["id"])
        second_comments = second_comments["results"]
        self.assertEqual(len(second_comments), 1)
        self.assertTrue(second_comments[0]["text"] == second_text)
        self.assertTrue(second_comments[0]["document_id"] == second_id)

    def test_that_comment_filter_returns_a_document_that_has_a_comment(self):
        document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
        text = "The number of seconds you shall wait is 3, not 2, 4 is totally out of question."
        self._add_comment_to_document(document["_id"], text, self.annotator["id"])
        url = reverse("v2:annotator-pull-commented", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})
        response = self.client.post(url)
        print_output("test_that_comment_filter_returns_a_document_that_has_a_comment:response.data", response.data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_that_facts_are_overwritten_with_new_selections(self):
        annotation_url = reverse("v2:annotator-annotate-multilabel", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})

        random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
        bar, foo = self.labelset["values"][0], self.labelset["values"][1]

        annotation_response = self.client.post(annotation_url, data={
            "document_id": random_document["_id"],
            "index": random_document["_index"],
            "labels": [bar, foo]
        }, format="json")
        self.assertEqual(annotation_response.status_code, status.HTTP_200_OK)

        annotation_response = self.client.post(annotation_url, data={
            "document_id": random_document["_id"],
            "index": random_document["_index"],
            "labels": [bar]
        }, format="json")

        ec = ElasticCore()
        document_in_elastic = ec.es.get(index=random_document["_index"], id=random_document["_id"])
        facts = document_in_elastic["_source"][TEXTA_TAGS_KEY]
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["str_val"], bar)

    def test_that_double_skipped_document_wont_be_counted(self):
        random_document = self.pull_random_document(self.project.pk, self.annotator["id"]).data
        skip = self.skip_document(random_document["_id"], random_document["_index"], self.project.pk, self.annotator["id"])
        first_count = Annotator.objects.get(pk=self.annotator["id"]).skipped
        skip = self.skip_document(random_document["_id"], random_document["_index"], self.project.pk, self.annotator["id"])
        second_count = Annotator.objects.get(pk=self.annotator["id"]).skipped
        self.assertEqual(first_count, second_count)

    def test_that_going_through_skipped_documents_is_rotating(self):
        random_document = self.pull_random_document(self.project.pk, self.annotator["id"], document_counter=1).data
        skip = self.skip_document(random_document["_id"], random_document["_index"], self.project.pk, self.annotator["id"])
        second_random = self.pull_random_document(self.project.pk, self.annotator["id"], document_counter=5).data
        second_skip = self.skip_document(second_random["_id"], second_random["_index"], self.project.pk, self.annotator["id"])

        # Since there is no skipped document with a counter of zero it should upgrade and pull the one with 1.
        first_skipped = self.pull_skipped_document(self.project.pk, self.annotator["id"], document_counter=0).data
        self.assertTrue(first_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] == 1)

        # Check that it fetches the next one since there is no skipped document with 2.
        next_counter = first_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] + 1
        second_skipped = self.pull_skipped_document(self.project.pk, self.annotator["id"], document_counter=next_counter).data
        self.assertTrue(second_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] == 5)

        # Since there are only two skipped documents, trying to go ahead of the latest one it should rotate back.
        next_counter = second_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] + 1
        third_skipped = self.pull_skipped_document(self.project.pk, self.annotator["id"], document_counter=next_counter).data
        self.assertTrue(third_skipped["_source"][TEXTA_ANNOTATOR_KEY]["document_counter"] == 1)
