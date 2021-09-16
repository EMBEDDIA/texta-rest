import uuid
import json
from django.test import override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITransactionTestCase
from toolkit.embedding.models import Embedding
from toolkit.elastic.tools.searcher import EMPTY_QUERY
from toolkit.elastic.tools.core import ElasticCore
from toolkit.core.task.models import Task
from toolkit.helper_functions import reindex_test_dataset
from toolkit.tools.utils_for_tests import create_test_user, project_creation, print_output
from toolkit.test_settings import (TEST_VERSION_PREFIX, VERSION_NAMESPACE, TEST_FIELD_CHOICE)


@override_settings(CELERY_ALWAYS_EAGER=True)
class RakunViewTest(APITransactionTestCase):
    def setUp(self):
        self.test_index_name = reindex_test_dataset()

        self.user = create_test_user('user', 'my@email.com', 'pw')
        self.project = project_creation("RakunExtractorTestProject", self.test_index_name, self.user)
        self.project.users.add(self.user)
        self.url = f'{TEST_VERSION_PREFIX}/projects/{self.project.id}/rakun_extractors/'
        self.embedding_url = f'{TEST_VERSION_PREFIX}/projects/{self.project.id}/embeddings/'

        self.rakun_id = None
        self.client.login(username='user', password='pw')

        """Create FastText Embedding, which will save facebook_model"""
        fasttext_payload = {
            "description": "TestEmbedding",
            "query": json.dumps(EMPTY_QUERY),
            "fields": TEST_FIELD_CHOICE,
            "max_vocab": 100,
            "min_freq": 5,
            "num_dimensions": 10,
            "embedding_type": "FastTextEmbedding"
        }
        print_output("Staring fasttext embedding", "doing post")

        response = self.client.post(self.embedding_url, json.dumps(fasttext_payload), content_type='application/json')
        print_output('test_create_fasttext_embedding_training_and_task_signal:response.data', response.data)
        # Check if Embedding gets created
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        created_embedding = Embedding.objects.get(id=response.data['id'])
        self.test_embedding_id = created_embedding.id
        # Remove Embedding files after test is done
        print_output("created fasttext embedding task status", created_embedding.task.status)
        # Check if Task gets created via a signal
        self.assertTrue(created_embedding.task is not None)
        # Check if Embedding gets trained and completed
        self.assertEqual(created_embedding.task.status, Task.STATUS_COMPLETED)

        ids = []
        payloads = [
            {
                "description": "test_all",
                "distance_method": "fasttext",
                "distance_threshold": 1,
                "num_keywords": 1,
                "pair_diff_length": 1,
                "stopwords": ["word1", "word2"],
                "bigram_count_threshold": 2,
                "min_tokens": 1,
                "max_tokens": 2,
                "max_similar": 2,
                "max_occurrence": 2,
                "fasttext_embedding": self.test_embedding_id
            },
            {
                "description": "rakun_test_new",
                "distance_method": "fasttext",
                "distance_threshold": 1.0,
                "min_tokens": 1,
                "max_tokens": 2,
                "fasttext_embedding": self.test_embedding_id
            }
        ]

        rakun_url = reverse(f"{VERSION_NAMESPACE}:rakun_extractor-list", kwargs={"project_pk": self.project.pk})
        for payload in payloads:
            response = self.client.post(rakun_url, payload)
            self.assertTrue(response.status_code == status.HTTP_201_CREATED)
            ids.append(int(response.data["id"]))

    def tearDown(self) -> None:
        ec = ElasticCore()
        ec.delete_index(index=self.test_index_name, ignore=[400, 404])
        print_output(f"Delete [Rakun Extractor] test index {self.test_index_name}", None)
        Embedding.objects.all().delete()
        ElasticCore().delete_index(index=self.test_index_name, ignore=[400, 404])
        print_output(f"Delete Rakun FASTTEXT Embeddings", None)

    def test(self):
        self.run_test_rakun_extractor_create()
        self.run_test_apply_rakun_extractor_to_index()

    def run_test_rakun_extractor_create(self):
        pass

    def run_test_apply_rakun_extractor_to_index(self):
        pass
