import pathlib
import os
import json
import uuid
from io import BytesIO
from time import sleep

from django.test import override_settings
from rest_framework import status
from rest_framework.test import APITransactionTestCase

from toolkit.elastic.tools.aggregator import ElasticAggregator
from toolkit.elastic.tools.core import ElasticCore

from toolkit.core.task.models import Task

from toolkit.test_settings import (
    TEST_FACT_NAME,
    TEST_FIELD_CHOICE,
    TEST_INDEX,
    TEST_VERSION_PREFIX,
    TEST_KEEP_PLOT_FILES,
    TEST_QUERY,
    )

from toolkit.tools.utils_for_tests import create_test_user, print_output, project_creation, remove_file, remove_folder
from toolkit.evaluator.models import Evaluator as EvaluatorObject



TEST_INDEX = "evaluator_multilabel_2k"

@override_settings(CELERY_ALWAYS_EAGER=True)
class EvaluatorObjectViewTests(APITransactionTestCase):
    def setUp(self):
        # Owner of the project

        self.user = create_test_user("EvaluatorOwner", "my@email.com", "pw")
        self.project = project_creation("EvaluatorTestProject", TEST_INDEX, self.user)
        self.project.users.add(self.user)
        self.url = f"{TEST_VERSION_PREFIX}/projects/{self.project.id}/evaluators/"
        self.project_url = f"{TEST_VERSION_PREFIX}/projects/{self.project.id}"

        self.test_binary_evaluator_id = None
        self.test_multilabel_evaluator_id = None

        self.true_fact_name = "TRUE_TAG"
        self.pred_fact_name = "PREDICTED_TAG"


        self.client.login(username="EvaluatorOwner", password="pw")


    def test(self):
        # to test: different averaging functions
        # multilabel
        # binary
        # multilabel with add_individual_results = True
        # multilabel with add_individual_results = False
        # negative:
        # fact not in the index
        # fact value not in the index
        # add memory buffer and test results with scrolling
        # individual_results view
        # individual_results view with restrictions (+ invalid restrisctions)
        # filetered_average view
        # filtered average view with restrictions
        self.run_smth()

        #pass


    def add_cleanup_files(self, evaluator_id: int):
        evaluator_object = EvaluatorObject.objects.get(pk=evaluator_id)
        if not TEST_KEEP_PLOT_FILES:
            self.addCleanup(remove_file, evaluator_object.plot.path)


    def run_smth(self):
        """ TODO"""
        payload = {
            "description": "Test Multilabel Evaluator",
            "indices": [{"name": TEST_INDEX}],
            "true_fact": self.true_fact_name,
            "predicted_fact": self.pred_fact_name,
            "average_function": "macro",
            "scroll_size": 500,
            "add_individual_results": True

        }
        response = self.client.post(self.url, payload, format="json")
        sleep(3)

        print_output("TODO:response.data", response.data)
        # Check if BertTagger gets created
        #self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        # Give the tagger some time to finish training
        evaluator_id = response.data["id"]
        evaluator_object = EvaluatorObject.objects.get(pk=evaluator_id)
        evaluator_json = evaluator_object.to_json()
        #evaluator_json.pop("binary_results")
        print_output("TODO:response.data", evaluator_json)

        self.add_cleanup_files(evaluator_id)
