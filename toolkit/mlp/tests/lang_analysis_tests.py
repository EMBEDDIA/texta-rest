import json

from django.test import override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITransactionTestCase

from toolkit.elastic.tools.searcher import ElasticSearcher
from toolkit.settings import NAN_TOKEN_KEY
from toolkit.test_settings import (TEST_FIELD, TEST_INDEX)
from toolkit.tools.utils_for_tests import create_test_user, project_creation


@override_settings(CELERY_ALWAYS_EAGER=True)
class ApplyLangViewsTests(APITransactionTestCase):

    def setUp(self) -> None:
        self.user = create_test_user('langDetectUser', 'my@email.com', 'pw')
        self.non_project_user = create_test_user('langDetectUserThatIsNotInProject', 'my@email.com', 'pw')
        self.project = project_creation("langDetectProject", TEST_INDEX, self.user)
        self.project.users.add(self.user)
        self.client.login(username='langDetectUser', password='pw')
        self.url = reverse("v2:lang_index-list", kwargs={"project_pk": self.project.pk})


    def test_unauthenticated_project_access(self):
        self.client.logout()
        self.client.login(username="langDetectUserThatIsNotInProject", password="pw")
        response = self.client.get(self.url)
        self.assertTrue(response.status_code == status.HTTP_403_FORBIDDEN)


    def test_unauthenticated_view_access(self):
        self.client.logout()
        response = self.client.get(self.url)
        self.assertTrue(response.status_code == status.HTTP_403_FORBIDDEN)


    def test_applying_lang_detect_with_query(self):
        mlp_field = f"{TEST_FIELD}_mlp"
        query_string = "inimene"
        payload = {
            "description": "TestingIndexProcessing",
            "field": TEST_FIELD,
            "query": json.dumps({'query': {'match': {'comment_content_lemmas': query_string}}}, ensure_ascii=False)
        }
        response = self.client.post(self.url, data=payload, format="json")
        self.assertTrue(response.status_code == status.HTTP_201_CREATED)
        s = ElasticSearcher(indices=[TEST_INDEX], output=ElasticSearcher.OUT_DOC, query=payload["query"])
        for hit in s:
            if TEST_FIELD in hit:
                self.assertTrue(f"{mlp_field}.language.detected" in hit)
                lang_value = hit[f"{mlp_field}.language.detected"]
                self.assertTrue(lang_value == "et" or lang_value == NAN_TOKEN_KEY)


    def test_applying_lang_detect_with_faulty_field_path(self):
        pass


    def test_with_non_existing_indices_in_payload(self):
        pass


    def test_that_lang_detect_enters_nan_token_on_bogus_fields(self):
        pass
