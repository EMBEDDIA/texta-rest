import json
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from toolkit.elastic.tools.document import ElasticDocument
from toolkit.tools.utils_for_tests import create_test_user, project_creation, print_output


class SearchFieldsTaggerIndexViewTests(APITestCase):

    @classmethod
    def setUpTestData(cls):
        cls.user = create_test_user('user', 'my@email.com', 'pw')
        cls.project = project_creation("SearchFieldsTaggerTestProject", "test_search_fields_tagger_index", cls.user)
        cls.project.users.add(cls.user)
        cls.url = reverse("v1:search_fields_tagger-list", kwargs={"project_pk": cls.project.pk})

        cls.uuid = "adasda-5874856a-das4das98f6"
        cls.document = {"Field_1": "This is sentence1. This is sentence2. This is sentence3. This is sentence4. This is sentence5.",
                        "Field_2": "This is a different sentence.",
                        "Field_3": "This is test data.",
                        "uuid": cls.uuid}

        cls.ed = ElasticDocument(index="test_search_fields_tagger_index")

        cls.ed.add(cls.document)

        cls.summarizer_id = None

    def setUp(self):
        self.client.login(username='user', password='pw')

    def tearDown(self) -> None:
        from toolkit.elastic.tools.core import ElasticCore
        ElasticCore().delete_index(index="test_search_fields_tagger_index", ignore=[400, 404])

    def test(self):
        self.run_test_search_fields_tagger()

    def run_test_search_fields_tagger(self):
        payload = {
                    "indices": [{"name": "test_search_fields_tagger_index"}],
                    "description": "test",
                    "query": json.dumps({}),
                    "fields": ["Field_1"],
                    "fact_name": "test_name"
                }

        response = self.client.post(self.url, payload)
        print_output('test_search_fields_tagger:response', response)

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
