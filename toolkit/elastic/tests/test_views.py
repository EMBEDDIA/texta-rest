import json
import os
from django.db.models import signals

from rest_framework.test import APITestCase
from rest_framework import status
from rest_framework.test import APIClient

from toolkit.test_settings import TEST_FIELD, TEST_INDEX, TEST_FIELD_CHOICE, TEST_INDEX_REINDEX
from toolkit.core.project.models import Project
from toolkit.elastic.models import Reindexer
from toolkit.elastic.core import ElasticCore
from toolkit.core.task.models import Task
from toolkit.tools.utils_for_tests import create_test_user, print_output, remove_file


class ReindexerViewTests(APITestCase):

    @classmethod
    def setUpTestData(cls):
        # Owner of the project
        cls.user = create_test_user('embeddingOwner', 'my@email.com', 'pw')

        cls.project = Project.objects.create(
            title='ReindexerTestProject',
            owner=cls.user,
            indices=TEST_INDEX
        )

        cls.url = f'/projects/{cls.project.id}/reindexer/'

    def setUp(self):
        self.client.login(username='embeddingOwner', password='pw')


    def test_run(self):
        self.run_create_reindexer_task_signal()


    def run_create_reindexer_task_signal(self):
        '''Tests the endpoint for a new Reindexer task, and if a new Task gets created via the signal'''
        payload = {
            "description": "TestReindexer",
            "fields": [TEST_FIELD],
            "indices": [TEST_INDEX],
            "new_index": TEST_INDEX_REINDEX
        }

        response = self.client.post(self.url, payload, format='json')
        print_output('run_create_reindexer_task_signal:response.data', response.data)
        # Check if new_index gets created
        # self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        created_reindexer = Reindexer.objects.get(id=response.data['id'])
        # self.test_reindexer_id = created_reindexer.id
        # Check if Embedding gets trained and completed
        # self.assertEqual(created_reindexer.task.status, Task.STATUS_COMPLETED)
        # remove test texta_test_index_reindexed
        new_index = response.data['new_index']
        ElasticCore().delete_index(new_index)
