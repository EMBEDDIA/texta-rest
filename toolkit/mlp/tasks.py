import json
import logging
from typing import List, Optional

from celery.decorators import task
from texta_mlp.mlp import MLP

from toolkit.base_tasks import BaseTask, TransactionAwareTask
from toolkit.elastic.document import ElasticDocument
from toolkit.elastic.searcher import ElasticSearcher
from toolkit.mlp.helpers import process_actions
from toolkit.mlp.models import MLPWorker
from toolkit.settings import CELERY_MLP_TASK_QUEUE, DEFAULT_MLP_LANGUAGE_CODES, INFO_LOGGER, MLP_MODEL_DIRECTORY
from toolkit.tools.show_progress import ShowProgress


# TODO Temporally as for now no other choice is found for sharing the models through the worker across the tasks.
mlp: Optional[MLP] = None


def load_mlp():
    global mlp
    if mlp is None:
        mlp = MLP(
            language_codes=DEFAULT_MLP_LANGUAGE_CODES,
            default_language_code="et",
            resource_dir=MLP_MODEL_DIRECTORY,
            logging_level="info"
        )


@task(name="apply_mlp_on_list", base=BaseTask, queue=CELERY_MLP_TASK_QUEUE, bind=True)
def apply_mlp_on_list(self, texts: List[str], analyzers: List[str]):
    load_mlp()
    response = []
    for text in texts:
        analyzed_text = mlp.process(text, analyzers)
        response.append(analyzed_text)
    return response


@task(name="apply_mlp_on_docs", base=BaseTask, queue=CELERY_MLP_TASK_QUEUE, bind=True)
def apply_mlp_on_docs(self, docs: List[dict], analyzers: List[str], fields_to_parse: List[str]):
    load_mlp()
    response = mlp.process_docs(docs=docs, analyzers=analyzers, doc_paths=fields_to_parse)
    return response


@task(name="start_mlp_worker", base=TransactionAwareTask, queue=CELERY_MLP_TASK_QUEUE, bind=True)
def start_mlp_worker(self, mlp_id: int):
    logging.getLogger(INFO_LOGGER).info(f"Starting applying mlp on the index for model ID: {mlp_id}")
    mlp_object = MLPWorker.objects.get(pk=mlp_id)
    show_progress = ShowProgress(mlp_object.task, multiplier=1)
    show_progress.update_step('running mlp')
    show_progress.update_view(0)

    return mlp_id


@task(name="apply_mlp_on_index", base=TransactionAwareTask, queue=CELERY_MLP_TASK_QUEUE, bind=True)
def apply_mlp_on_index(self, mlp_id: int):
    load_mlp()
    mlp_object = MLPWorker.objects.get(pk=mlp_id)
    show_progress = ShowProgress(mlp_object.task, multiplier=1)
    show_progress.update_step('scrolling mlp')

    # Get the necessary fields.
    indices: List[str] = mlp_object.get_indices()
    field_data: List[str] = json.loads(mlp_object.fields)
    analyzers: List[str] = json.loads(mlp_object.analyzers)

    searcher = ElasticSearcher(
        query=mlp_object.query,
        indices=indices,
        field_data=field_data,
        output=ElasticSearcher.OUT_RAW,
        callback_progress=show_progress,
        scroll_size=100,
        scroll_timeout="30m"
    )

    for index in indices:
        searcher.core.add_texta_facts_mapping(index=index)

    actions = process_actions(searcher, analyzers, field_data, mlp=mlp)

    # Send the data towards Elasticsearch
    ed = ElasticDocument("_all")
    elastic_response = ed.bulk_update(actions=actions)

    return mlp_id


@task(name="end_mlp_task", base=TransactionAwareTask, queue=CELERY_MLP_TASK_QUEUE, bind=True)
def end_mlp_task(self, mlp_id):
    logging.getLogger(INFO_LOGGER).info(f"Finished applying mlp on the index for model ID: {mlp_id}")
    mlp_object = MLPWorker.objects.get(pk=mlp_id)
    mlp_object.task.complete()
    return True
