from rest_framework.test import APITestCase
from toolkit.tools.utils_for_tests import create_test_user, project_creation, print_output
from toolkit.test_settings import TEST_INDEX, TEST_VERSION_PREFIX
from rest_framework import status


class SummarizerIndexViewTests(APITestCase):

    @classmethod
    def setUpTestData(cls):
        cls.user = create_test_user('user', 'my@email.com', 'pw')
        cls.project = project_creation("SummarizerTestProject", TEST_INDEX, cls.user)
        cls.project.users.add(cls.user)
        cls.url = f'{TEST_VERSION_PREFIX}/projects/{cls.project.id}/summarizer_index/'

        cls.summarizer_id = None

    def setUp(self):
        self.client.login(username='user', password='pw')

    def test(self):
        self.run_test_summarizer_create()

    def run_test_summarizer_create(self):
        payload = {
            "description": "TestSummarizer",
            "query": {},
            "fields": ["Field_1"]
        }

        response = self.client.post(self.url, payload)
        print_output('test_summarizer_create:response.data', response.data)
        created_id = response.data['id']

        self.summarizer_id = created_id

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
