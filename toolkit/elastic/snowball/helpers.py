import logging

from toolkit.elastic.tools.searcher import ElasticSearcher
from toolkit.settings import INFO_LOGGER


def process_stemmer_actions(generator: ElasticSearcher, worker):
    counter = 0
    info_logger = logging.getLogger(INFO_LOGGER)

    info_logger.info(f"Applying stemming to the worker with an ID of {worker.pk}!")
    for document_batch in generator:
        for item in document_batch:
            # This will be a list of texts.
            source = item["_source"]

            elastic_update_body = {
                "_id": item["_id"],
                "_index": item["_index"],
                "_type": item.get("_type", "_doc"),
                "_op_type": "update",
                "retry_on_conflict": 3,
                "doc": {**source}
            }

            yield elastic_update_body

            counter += 1
            progress = generator.callback_progress
            if counter % generator.scroll_size == 0:
                info_logger.info(f"Progress on applying language detection for worker with id: {worker.pk} at {counter} out of {progress.n_total} documents!")
            elif counter == progress.n_total:
                info_logger.info(f"Finished applying language detection for worker with id: {worker.pk} at {counter}/{progress.n_total} documents!")
