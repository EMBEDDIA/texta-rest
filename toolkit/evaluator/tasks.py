import json
import os
import secrets
import logging
import pathlib
import psutil

import numpy as np
from celery import group
from celery.decorators import task
from celery.result import allow_join_result
from django.db import connections
from elasticsearch.helpers import streaming_bulk

from toolkit.core.task.models import Task

from toolkit.evaluator.models import Evaluator
from toolkit.evaluator import choices
from toolkit.base_tasks import TransactionAwareTask, BaseTask

from toolkit.elastic.tools.data_sample import DataSample
from toolkit.elastic.tools.feedback import Feedback
from toolkit.elastic.tools.searcher import ElasticSearcher
from toolkit.elastic.tools.core import ElasticCore
from toolkit.elastic.tools.document import ElasticDocument
from toolkit.elastic.tools.aggregator import ElasticAggregator
from toolkit.elastic.tools.query import Query

from toolkit.tools.show_progress import ShowProgress
from toolkit.settings import CELERY_LONG_TERM_TASK_QUEUE, CELERY_SHORT_TERM_TASK_QUEUE, INFO_LOGGER, ERROR_LOGGER, MEDIA_URL
from toolkit.helper_functions import get_core_setting, get_indices_from_object
from toolkit.tools.plots import create_confusion_plot

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from collections import defaultdict
from typing import List, Union, Dict, Tuple
from copy import deepcopy


def filter_results(binary_results: dict, min_count: int, max_count: int, metric_restrictions: json) -> json:
    """ Filter multilabel scores based on label count and metric scores restrictions. """
    filtered_scores = {}

    for label, label_scores in list(binary_results.items()):
        count = np.sum(np.array(label_scores["confusion_matrix"]))

        if min_count <= count <= max_count:

            # Add label count to the output
            label_scores["count"] = count

            filters_passed = True
            print(metric_restrictions)
            # Check if the label scores pass all restrictions
            for metric, restrictions in list(metric_restrictions.items()):
                if ("min_score" in restrictions and label_scores[metric] < restrictions["min_score"]) or \
                   ("max_score" in restrictions and label_scores[metric] > restrictions["max_score"]):
                   filters_passed = False
                   break

            if filters_passed:
                filtered_scores[label] = label_scores
    return filtered_scores


def filter_and_average_results(binary_results: dict, min_count: int, max_count: int, metric_restrictions: json) -> json:
    """ Calculate average of filtered tag scores. """
    metrics = choices.METRICS
    filtered_scores = {metric: [] for metric in metrics}

    for label, label_scores in list(binary_results.items()):
        count = np.sum(np.array(label_scores["confusion_matrix"]))

        # If label count is in required range, add scores to corresponding lists
        if min_count <= count <= max_count:
            filters_passed = True

            # Check if the label scores pass all restrictions
            for metric, restrictions in list(metric_restrictions.items()):
                if ("min_score" in restrictions and label_scores[metric] < restrictions["min_score"]) or \
                   ("max_score" in restrictions and label_scores[metric] > restrictions["max_score"]):
                   filters_passed = False
                   break

            if filters_passed:
                for metric in metrics:
                    filtered_scores[metric].append(label_scores[metric])

    # Calculate average scores of filtered scores
    avg_scores = {}
    for metric in metrics:
        avg_scores[metric] = np.mean(filtered_scores[metric])
    avg_scores["count"] = len(filtered_scores[metric])
    return avg_scores


def get_memory_imprint(n_docs: int, n_classes: int, eval_type: str, unit="gb", int_size: int=64) -> int:
    """ Get required memory space for 2 matrices with size (n_docs, n_classes)
    and dtype = int{int_size}.
    """
    unit_map = {"gb": 1024**3, "mb": 1024**2, "kb": 1024**1, "b": 1024**0}
    matrices_imprint = 2*((n_docs*n_classes*(int_size/2**3))/unit_map[unit])
    classes_imprint = (n_classes*(int_size/2**3))/unit_map[unit]
    if eval_type == "binary":
        total_imprint = matrices_imprint + classes_imprint
    else:
        # For mutlilabel and multiclass evaluation, individial scores are stored as well
        # taking the same amount of space as
        binary_imprint = 2*((n_docs*(int_size/2**3))/unit_map[unit])
        total_imprint = matrices_imprint + binary_imprint + classes_imprint
    return total_imprint


def is_enough_memory_available(required_memory: float, memory_buffer: float, unit="gb"):
    """ Checks if the system has enough memory for the task."""
    unit_map = {"gb": 1024**3, "mb": 1024**2, "kb": 1024**1, "b": 1024**0}
    available_memory = (psutil.virtual_memory().available / unit_map[unit]) - memory_buffer

    logging.getLogger(INFO_LOGGER).info(f"Required memory: {required_memory}{unit.upper()}\nMemory buffer: {memory_buffer}{unit.upper()}\nAvailable memory: {available_memory}{unit.upper()}")
    return available_memory >= required_memory


def get_facts_by_name(texta_facts: List[dict], fact_name: str):
    """ Returns list of fact values corresponding to `fact_name`. """
    return [fact["str_val"] for fact in texta_facts if fact["fact"] == fact_name]


def get_scores(true_labels: List[Union[str, int]], pred_labels: List[Union[str, int]], classes: List[str], average: str) -> dict:
    """ Calculate different metrics" scores with sklearn. """

    bin_scores = {}
    if len(classes) > 2:
        # Binarize multilabel results
        mlb = MultiLabelBinarizer(classes=classes)
        true_labels = mlb.fit_transform(true_labels).astype("int8")
        pred_labels = mlb.fit_transform(pred_labels).astype("int8")

        confusion_classes = [i for i in range(len(classes))]

        if len(classes) <= choices.DEFAULT_MAX_CONFUSION_CLASSES:
            confusion = confusion_matrix(true_labels.argmax(axis=1), pred_labels.argmax(axis=1), labels=confusion_classes)#.tolist()
        else:
            confusion = np.array([[]])

        for i, label_class in enumerate(classes):
            label_scores = get_scores(true_labels[:, i], pred_labels[:, i], classes=[0, 1], average="binary")
            label_scores.pop("bin_scores")
            bin_scores[label_class] = label_scores

    else:
        # Use numerical classes for binary taggers to avoid conflicts
        # when calculating confusion matrix
        classes = [0, 1]
        confusion = confusion_matrix(true_labels, pred_labels, labels=classes)#.tolist()
        print("Confusion type", type(confusion), "data type", confusion.dtype)

    #logging.getLogger(INFO_LOGGER).info(f"TRUE_LABLS: {true_labels}\nPRED_LABELS: {pred_labels}")

    scores = {
        "precision": precision_score(true_labels, pred_labels, average=average),
        "recall": recall_score(true_labels, pred_labels, average=average),
        "f1_score": f1_score(true_labels, pred_labels, average=average),
        "accuracy": accuracy_score(true_labels, pred_labels),
        "confusion_matrix": confusion,
        "bin_scores": bin_scores
    }

    return scores


def scroll_labels(generator: ElasticSearcher, true_fact: str, pred_fact: str, true_fact_value: str = "", pred_fact_value: str = "", classes: List[Union[str, int]]=[], average: str = "macro", score_after_scroll: bool = True) -> Tuple[Union[List[int], List[List[str]]], Union[List[int], List[List[str]]], Dict[str, Dict[str, List[int]]]]:
    true_labels = []
    pred_labels = []
    if len(classes) <= choices.DEFAULT_MAX_CONFUSION_CLASSES:
        empty_confusion_matrix = np.zeros((len(classes), len(classes)))
    else:
        empty_confusion_matrix = np.array([[]])

    """"scores = {
        "precision": [],
        "recall": [],
        "f1_score": [],
        "accuracy": [],
        "confusion_matrix": empty_confusion_matrix
    }"""
    scores = {"confusion_matrix": empty_confusion_matrix}
    #bin_labels = defaultdict(lambda: defaultdict(list))
    bin_scores = defaultdict(dict)

    for i, scroll_batch in enumerate(generator):
        if score_after_scroll:
            true_labels = []
            pred_labels = []
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

                # Get binary
                """
                total_facts = list(set(true_facts).union(set(pred_facts)))

                for fact in total_facts:
                    true_label_i = 1 if fact in true_facts else 0
                    pred_label_i = 1 if fact in pred_facts else 0

                    bin_labels[fact]["true_labels"].append(true_label_i)
                    bin_labels[fact]["pred_labels"].append(pred_label_i)"""

        if score_after_scroll:
            batch_scores = get_scores(true_labels, pred_labels, classes, average)
            bin_batch_scores = batch_scores.pop("bin_scores")
            for metric in batch_scores.keys():
                if metric == "confusion_matrix":
                    scores[metric]+=batch_scores[metric]
                else:
                    if metric not in scores:
                        scores[metric] = batch_scores[metric]
                    else:
                        scores[metric] = np.mean([scores[metric], batch_scores[metric]])

            if bin_batch_scores:
                for label_class in classes:
                    for metric in batch_scores.keys():
                        if metric == "confusion_matrix":
                            if metric not in bin_scores[label_class]:
                                bin_scores[label_class][metric] = bin_batch_scores[label_class][metric]
                            else:
                                bin_scores[label_class][metric]+=bin_batch_scores[label_class][metric]

                        else:
                            if metric not in bin_scores:
                                bin_scores[label_class][metric] = bin_batch_scores[label_class][metric]
                            else:
                                bin_scores[label_class][metric] = np.mean([bin_scores[label_class][metric], bin_batch_scores[label_class][metric]])

                    #scores[metric].append(batch_scores[metric])

    if not score_after_scroll:
        batch_scores = get_scores(true_labels, pred_labels, classes, average)
        bin_scores = batch_scores.pop("bin_scores")
        for metric in batch_scores.keys():
            if metric == "confusion_matrix":
                scores[metric]+=batch_scores[metric]
            else:
                if metric not in scores:
                    scores[metric] = batch_scores[metric]
                else:
                    scores[metric] = np.mean([scores[metric], batch_scores[metric]])

    # convert bin labels confusion matrix from numpy array to list
    for label in bin_scores:
        bin_scores[label]["confusion_matrix"] = bin_scores[label]["confusion_matrix"].astype("int").tolist()

    return (scores, bin_scores)




@task(name="evaluate_tags", base=TransactionAwareTask, queue=CELERY_LONG_TERM_TASK_QUEUE)
def evaluate_tags_task(object_id: int, indices: List[str], query: dict, es_timeout: int = 10, scroll_size: int = 100):
    try:
        logging.getLogger(INFO_LOGGER).info(f"Starting evaluator task for Evaluator with ID {object_id}.")

        evaluator_object = Evaluator.objects.get(pk=object_id)
        progress = ShowProgress(evaluator_object.task)

        true_fact = evaluator_object.true_fact
        pred_fact = evaluator_object.predicted_fact
        true_fact_value = evaluator_object.true_fact_value
        pred_fact_value = evaluator_object.predicted_fact_value

        average = evaluator_object.average_function

        q = Query()

        fact_names = [true_fact, pred_fact]
        fact_values = [true_fact_value, pred_fact_value]

        q.add_facts_filter(fact_names, fact_values, operator="should", max_inner_hits=100)

        query = q.__dict__()

        searcher = ElasticSearcher(
            indices = indices,
            field_data = ["texta_facts"],
            query = query,
            output = ElasticSearcher.OUT_RAW,
            timeout = f"{es_timeout}m",
            callback_progress=progress,
            scroll_size = scroll_size
        )

        #n_docs = searcher.count()
        print("QUERY B4 AGGS", searcher.query)




        # Binary
        if true_fact_value and pred_fact_value:
            logging.getLogger(INFO_LOGGER).info(f"Starting binary evaluation. Comparing following fact and fact value pairs: [{true_fact}: {true_fact_value}] and [{pred_fact}: {pred_fact_value}].")
            classes = ["other", true_fact_value]
            evaluator_object.evaluation_type = "binary"
            true_set = {true_fact_value, "other"}
            pred_set = {pred_fact_value, "other"}


        # Multilabel
        else:
            logging.getLogger(INFO_LOGGER).info(f"Starting multilabel evaluation. Comparing fact values of facts '{true_fact}' and '{pred_fact}'.")
            # Make deepcopy of the query to avoid modifying Searcher's query.
            es_aggregator = ElasticAggregator(indices = indices, query = deepcopy(query))

            # TODO: does the aggregation need size update?
            true_fact_values = es_aggregator.facts(size=choices.DEFAULT_MAX_AGGREGATION_SIZE, filter_by_fact_name=true_fact)
            pred_fact_values = es_aggregator.facts(size=choices.DEFAULT_MAX_AGGREGATION_SIZE, filter_by_fact_name=pred_fact)

            true_set = set(true_fact_values)
            pred_set = set(pred_fact_values)

            classes = list(true_set.union(pred_set))
            evaluator_object.evaluation_type = "multilabel"

        #stats = searcher.core.get_index_stats(indices)
        #print("INDEX STATS", stats)
        #n_docs = 0
        #for index in indices:
        #    n_docs+=stats[index]["doc_count"]
        print("QUERY AFTER AGGS", searcher.query)
        n_docs = searcher.count()
        logging.getLogger(INFO_LOGGER).info(f"Number of documents: {n_docs}\nNumber of classes: {len(classes)}")

        required_memory = get_memory_imprint(n_docs=n_docs, n_classes=len(classes), eval_type=evaluator_object.evaluation_type, unit="gb", int_size=64)
        enough_memory = is_enough_memory_available(required_memory=required_memory, memory_buffer=choices.DEFAULT_MEMORY_BUFFER_GB, unit="gb")
        score_after_scroll = False if enough_memory else True

        scores_imprecise = True if (score_after_scroll and average != "micro") else False

        logging.getLogger(INFO_LOGGER).info(f"Enough available memory: {enough_memory}\nScore after scroll: {score_after_scroll}")


        scores, bin_scores = scroll_labels(generator=searcher, true_fact = true_fact, pred_fact = pred_fact, true_fact_value = true_fact_value, pred_fact_value=pred_fact_value, classes=classes, average=average, score_after_scroll=score_after_scroll)
        #scores = get_scores(true_labels, pred_labels, classes, average)
        logging.getLogger(INFO_LOGGER).info(f"Scores per batch: {scores}")


        """for metric in scores:
            if metric != "confusion_matrix":
                scores[metric] = np.mean(scores[metric])
            else:
                # convert matrix values back to int
                scores[metric] = scores[metric].astype("int64")

        bin_scores = {}

        # Calculate individual results for multilabel tags
        if not true_fact_value and not pred_fact_value:
            logging.getLogger(INFO_LOGGER).info(f"Starting individual evaluation of multilabel labels.")
            for label in bin_labels:
                true_labels = bin_labels[label]["true_labels"]
                pred_labels = bin_labels[label]["pred_labels"]
                label_scores = get_scores(true_labels, pred_labels, [0, 1], "binary")
                label_scores["confusion_matrix"] = label_scores["confusion_matrix"].tolist()
                bin_scores[label] = label_scores"""

        # Add confusion classes to the output + confusion plot?
        # Save the image before its path.
        image_name = f'{secrets.token_hex(15)}.png'
        evaluator_object.plot.save(image_name, create_confusion_plot(scores["confusion_matrix"], classes), save=False)
        image_path = pathlib.Path(MEDIA_URL) / image_name

        evaluator_object.document_count = n_docs

        evaluator_object.scores_imprecise = scores_imprecise
        evaluator_object.precision = scores["precision"]
        evaluator_object.recall = scores["recall"]
        evaluator_object.f1_score = scores["f1_score"]
        evaluator_object.accuracy = scores["accuracy"]
        evaluator_object.confusion_matrix = json.dumps(scores["confusion_matrix"].astype("int").tolist())
        evaluator_object.n_true_classes = len(true_set)
        evaluator_object.n_predicted_classes = len(pred_set)
        evaluator_object.n_total_classes = len(classes)
        evaluator_object.binary_scores = json.dumps(bin_scores)
        evaluator_object.plot.name = str(image_path)

        evaluator_object.save()
        evaluator_object.task.complete()
        return True

    except Exception as e:
        logging.getLogger(ERROR_LOGGER).exception(e)
        error_message = f"{str(e)[:100]}..."  # Take first 100 characters in case the error message is massive.
        evaluator_object.task.add_error(error_message)
        evaluator_object.task.fail()
