import json
import os
import secrets
import logging
from celery.decorators import task
from django.db import connections
from elasticsearch.helpers import streaming_bulk

from toolkit.core.task.models import Task

from toolkit.evaluator.models import Evaluator
from toolkit.base_tasks import TransactionAwareTask
from toolkit.elastic.data_sample import DataSample
from toolkit.elastic.feedback import Feedback
from toolkit.elastic.searcher import ElasticSearcher
from toolkit.elastic.core import ElasticCore
from toolkit.elastic.document import ElasticDocument
from toolkit.elastic.aggregator import ElasticAggregator

from toolkit.tools.show_progress import ShowProgress
from toolkit.settings import CELERY_LONG_TERM_TASK_QUEUE, INFO_LOGGER, ERROR_LOGGER
from toolkit.helper_functions import get_core_setting, get_indices_from_object

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from typing import List, Union, Dict, Tuple


def get_facts_by_name(texta_facts: List[dict], fact_name: str):
    """ Returns list of fact values corresponding to `fact_name`. """
    return [fact["str_val"] for fact in texta_facts if fact["fact"] == fact_name]


def scroll_labels(generator: ElasticSearcher, true_fact: str, pred_fact: str, true_fact_value: str = "", pred_fact_value: str = "") -> Tuple[List[str], List[str]]:
    true_labels = []
    pred_labels = []
    for i, scroll_batch in enumerate(generator):
        logging.getLogger(INFO_LOGGER).info(f"Evaluating batch {i+1}...")

        for raw_doc in scroll_batch:
            hit = raw_doc["_source"]
            facts = hit.get("texta_facts", [])

            true_facts = get_facts_by_name(facts, true_fact)
            pred_facts = get_facts_by_name(facts, pred_fact)

            # Binary evaluation
            if true_fact_value and pred_fact_value:
                true_label_i = 1 if true_fact_value in true_facts else 0
                pred_label_i = 1 if pred_fact_value in pred_facts else 0

                true_labels.append(true_label_i)
                pred_labels.append(pred_label_i)

            # Multilabel evaluation
            else:
                true_labels.append(true_facts)
                pred_labels.append(pred_facts)


    return (true_labels, pred_labels)


def get_scores(true_labels: List[Union[str, int]], pred_labels: List[Union[str, int]], classes: List[str], average: str) -> dict:
    """ Calculate different metrics' scores with sklearn. """
    # Binarize multilabel results
    if len(classes) > 2:
        mlb = MultiLabelBinarizer(classes=classes)
        true_labels = mlb.fit_transform(true_labels)
        pred_labels = mlb.fit_transform(pred_labels)
    else:
        # Use numerical classes for binary taggers to avoid conflicts
        # when calculating confusion matrix
        classes = [0, 1]

    logging.getLogger(INFO_LOGGER).info(f"TRUE_LABLS: {true_labels}\nPRED_LABELS: {pred_labels}")

    scores = {
        "precision": precision_score(true_labels, pred_labels, average=average),
        "recall": recall_score(true_labels, pred_labels, average=average),
        "f1_score": f1_score(true_labels, pred_labels, average=average),
        "accuracy": accuracy_score(true_labels, pred_labels),
        "confusion_matrix": confusion_matrix(true_labels, pred_labels, labels=classes).tolist()
    }

    return scores

@task(name="evaluate_tags", base=TransactionAwareTask, queue=CELERY_LONG_TERM_TASK_QUEUE)
def evaluate_tags_task(object_id: int, indices: List[str], query: dict, es_timeout: int = 10, bulk_size: int = 100):

    try:
        logging.getLogger(INFO_LOGGER).info(f"Starting evaluator task for Evaluator with ID {object_id}.")

        evaluator_object = Evaluator.objects.get(pk=object_id)
        progress = ShowProgress(evaluator_object.task)

        searcher = ElasticSearcher(
            indices = indices,
            field_data = ["texta_facts"],
            query = query,
            output = ElasticSearcher.OUT_RAW,
            timeout = f"{es_timeout}m",
            callback_progress=progress,
            scroll_size = bulk_size
        )


        true_fact = evaluator_object.true_fact
        pred_fact = evaluator_object.predicted_fact
        true_fact_value = evaluator_object.true_fact_value
        pred_fact_value = evaluator_object.predicted_fact_value

        average = evaluator_object.average_function

        # Binary
        if true_fact_value and pred_fact_value:
            logging.getLogger(INFO_LOGGER).info(f"Comparing facts [{true_fact}: {true_fact_value}] (TRUE) and [{pred_fact}: {pred_fact_value}] (PRED).")
            classes = [true_fact_value, pred_fact_value]


        # Multilabel
        else:
            logging.getLogger(INFO_LOGGER).info(f"Comparing facts {true_fact} (TRUE) and {pred_fact} (PRED).")
            es_aggregator = ElasticAggregator(indices = indices, query = query)

            # TODO: does the aggregation need size update?
            true_fact_values = es_aggregator.get_fact_values_distribution(true_fact)
            pred_fact_values = es_aggregator.get_fact_values_distribution(pred_fact)

            true_set = set(true_fact_values.keys())
            pred_set = set(pred_fact_values.keys())

            classes = list(true_set.union(pred_set))


        true_labels, pred_labels = scroll_labels(generator=searcher, true_fact = true_fact, pred_fact = pred_fact, true_fact_value = true_fact_value, pred_fact_value=pred_fact_value)
        scores = get_scores(true_labels, pred_labels, classes, average)

        evaluator_object.precision = scores["precision"]
        evaluator_object.recall = scores["recall"]
        evaluator_object.f1_score = scores["f1_score"]
        evaluator_object.accuracy = scores["accuracy"]
        evaluator_object.confusion_matrix = json.dumps(scores["confusion_matrix"])

        evaluator_object.save()

        evaluator_object.task.complete()
        return True

    except Exception as e:
        logging.getLogger(ERROR_LOGGER).exception(e)
        error_message = f"{str(e)[:100]}..."  # Take first 100 characters in case the error message is massive.
        evaluator_object.task.add_error(error_message)
