import json
import logging
from typing import List, Optional

from celery.decorators import task
from texta_mlp.mlp import MLP

from toolkit.core.task.models import Task
from toolkit.base_tasks import BaseTask, TransactionAwareTask
from toolkit.elastic.tools.document import ElasticDocument
from toolkit.elastic.tools.searcher import ElasticSearcher
from toolkit.mlp.helpers import process_lang_actions, process_mlp_actions
from toolkit.mlp.models import ApplyLangWorker, MLPWorker
from toolkit.settings import CELERY_MLP_TASK_QUEUE, DEFAULT_MLP_LANGUAGE_CODES, INFO_LOGGER, ERROR_LOGGER, MLP_MODEL_DIRECTORY
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
    mlp_object = MLPWorker.objects.get(pk=mlp_id)
    task_object = mlp_object.task
    try:
        load_mlp()
        show_progress = ShowProgress(task_object, multiplier=1)
        show_progress.update_step('scrolling mlp')

        # Get the necessary fields.
        indices: List[str] = mlp_object.get_indices()
        field_data: List[str] = json.loads(mlp_object.fields)
        analyzers: List[str] = json.loads(mlp_object.analyzers)
        es_scroll_size: int = mlp_object.es_scroll_size
        es_timeout: int = mlp_object.es_timeout

        searcher = ElasticSearcher(
            query=json.loads(mlp_object.query),
            indices=indices,
            field_data=field_data,
            output=ElasticSearcher.OUT_RAW,
            callback_progress=show_progress,
            scroll_size=es_scroll_size,
            scroll_timeout=f"{es_timeout}m"
        )

        for index in indices:
            searcher.core.add_texta_facts_mapping(index=index)

        actions = process_mlp_actions(searcher, analyzers, field_data, mlp_class=mlp, mlp_id=mlp_id)

        # Send the data towards Elasticsearch
        ed = ElasticDocument("_all")
        elastic_response = ed.bulk_update(actions=actions)
        return mlp_id

    except Exception as e:
        logging.getLogger(ERROR_LOGGER).exception(e)
        task_object.add_error(str(e))
        task_object.update_status(Task.STATUS_FAILED)
        raise e


@task(name="end_mlp_task", base=TransactionAwareTask, queue=CELERY_MLP_TASK_QUEUE, bind=True)
def end_mlp_task(self, mlp_id):
    logging.getLogger(INFO_LOGGER).info(f"Finished applying mlp on the index for model ID: {mlp_id}")
    mlp_object = MLPWorker.objects.get(pk=mlp_id)
    mlp_object.task.complete()
    return True


@task(name="apply_lang_on_indices", base=TransactionAwareTask, queue=CELERY_MLP_TASK_QUEUE, bind=True)
def apply_lang_on_indices(self, apply_worker_id: int):
    worker_object = ApplyLangWorker.objects.get(pk=apply_worker_id)
    task_object = worker_object.task
    try:
        load_mlp()
        show_progress = ShowProgress(task_object, multiplier=1)
        show_progress.update_step('scrolling through the indices to apply lang')

        # Get the necessary fields.
        indices: List[str] = worker_object.get_indices()
        field = worker_object.field

        scroll_size = 100
        searcher = ElasticSearcher(
            query=json.loads(worker_object.query),
            indices=indices,
            field_data=[field],
            output=ElasticSearcher.OUT_RAW,
            callback_progress=show_progress,
            scroll_size=scroll_size,
            scroll_timeout="15m"
        )

        for index in indices:
            searcher.core.add_texta_facts_mapping(index=index)

        actions = process_lang_actions(generator=searcher, field=field, worker_id=apply_worker_id, mlp_class=mlp)

        # Send the data towards Elasticsearch
        ed = ElasticDocument("_all")
        elastic_response = ed.bulk_update(actions=actions)

        worker_object.task.complete()

        return apply_worker_id

    except Exception as e:
        logging.getLogger(ERROR_LOGGER).exception(e)
        task_object.add_error(str(e))
        task_object.update_status(Task.STATUS_FAILED)
        raise e
