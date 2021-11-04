# Create your tests here.
import json

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from toolkit.annotator.models import Annotator
from toolkit.elastic.index.models import Index
from toolkit.elastic.tools.core import ElasticCore
from toolkit.helper_functions import reindex_test_dataset
from toolkit.test_settings import TEST_FIELD, TEST_MATCH_TEXT, TEST_QUERY
from toolkit.tools.utils_for_tests import create_test_user, print_output, project_creation


class BinaryAnnotatorTests(APITestCase):

    def setUp(self):
        # Owner of the project
        self.test_index_name = reindex_test_dataset()
        self.secondary_index = reindex_test_dataset()
        self.index, is_created = Index.objects.get_or_create(name=self.secondary_index)
        self.user = create_test_user('annotator', 'my@email.com', 'pw')
        self.project = project_creation("taggerTestProject", self.test_index_name, self.user)
        self.project.indices.add(self.index)
        self.project.users.add(self.user)

        self.client.login(username='annotator', password='pw')
        self.ec = ElasticCore()

        self.list_view_url = reverse("v2:annotator-list", kwargs={"project_pk": self.project.pk})
        self.annotator = self._create_annotator()
        self.pull_document_url = reverse("v2:annotator-pull-document", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})


    def test_all(self):
        self.run_pulling_document()
        self.run_binary_annotation()
        self.run_that_query_limits_pulled_document()
        doc_id_with_comment = self.run_adding_comment_to_document()
        self.run_pulling_comment_for_document(doc_id_with_comment)
        self.run_annotating_to_the_end()


    def _create_annotator(self):
        payload = {
            "description": "Random test annotation.",
            "indices": [{"name": self.test_index_name}, {"name": self.secondary_index}],
            "query": json.dumps(TEST_QUERY),
            "fields": ["comment_content", TEST_FIELD],
            "target_field": "comment_content",
            "annotation_type": "binary",
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
        self.assertTrue(total_count > response.data["total"])
        return response.data


    def _pull_random_document(self):
        url = reverse("v2:annotator-pull-document", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})
        response = self.client.post(url, format="json")
        return response.data


    def run_binary_annotation(self):
        annotation_url = reverse("v2:annotator-annotate-binary", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})

        random_document = self._pull_random_document()
        document_id = random_document["_id"]
        index = random_document["_index"]
        annotation_payloads = [
            {
                "annotation_type": "pos",
                "document_id": document_id,
                "index": index
            },
            {
                "annotation_type": "neg",
                "document_id": document_id,
                "index": index
            }
        ]

        for index_count, payload in enumerate(annotation_payloads):
            annotation_response = self.client.post(annotation_url, data=payload, format="json")
            # Test for response success.
            print_output("run_binary_annotation:response.status", annotation_response.status_code)
            self.assertTrue(annotation_response.status_code == status.HTTP_200_OK)

            # Test that progress is updated properly.
            model_object = Annotator.objects.get(pk=self.annotator["id"])
            self.assertTrue(model_object.annotated == index_count + 1)

            # Check that document was actually edited.
            es_doc = self.ec.es.get(index=index, id=document_id)["_source"]
            facts = es_doc["texta_facts"]
            self.assertTrue(model_object.binary_configuration.fact_name in [fact["fact"] for fact in facts])
            if payload["annotation_type"] == "pos":
                self.assertTrue(model_object.binary_configuration.pos_value in [fact["str_val"] for fact in facts])
            elif payload["annotation_type"] == "neg":
                self.assertTrue(model_object.binary_configuration.neg_value in [fact["str_val"] for fact in facts])


    def run_annotating_to_the_end(self):
        model_object = Annotator.objects.get(pk=self.annotator["id"])
        total = model_object.total
        annotated = model_object.annotated

        annotation_url = reverse("v2:annotator-annotate-binary", kwargs={"project_pk": self.project.pk, "pk": self.annotator["id"]})

        for i in range(total - annotated + 1):
            random_document = self._pull_random_document()
            payload = {"annotation_type": "pos", "document_id": random_document["_id"], "index": random_document["_index"]}
            annotation_response = self.client.post(annotation_url, data=payload, format="json")
            self.assertTrue(annotation_response.status_code == status.HTTP_200_OK)

        # At this point all the documents should be done.
        random_document = self._pull_random_document()
        self.assertTrue(random_document["detail"] == 'No more documents left!')


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
        random_document = self._pull_random_document()
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


    def run_that_query_limits_pulled_document(self):
        random_document = self._pull_random_document()
        content = random_document["_source"]
        print_output("run_that_query_limits_pulled_document:source", content)
        self.assertTrue(TEST_MATCH_TEXT in content.get(TEST_FIELD, ""))


    def run_that_double_skipped_document_wont_be_counted(self):
        pass
