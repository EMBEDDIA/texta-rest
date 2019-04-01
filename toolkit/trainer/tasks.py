from celery.decorators import task
from gensim.models import word2vec
import json
import os

from toolkit.trainer.models import Embedding, Task, Tagger
from toolkit.settings import NUM_WORKERS, MODELS_DIR
from toolkit.trainer.embedding.phraser import Phraser
from toolkit.elastic.searcher import ElasticSearcher
from toolkit.tools.show_progress import ShowProgress
from toolkit.trainer.tagger.text_tagger import TextTagger
from toolkit.tools.text_processor import TextProcessor

@task(name="train_embedding")
def train_embedding(embedding_id):
    # retrieve embedding & task objects
    embedding_object = Embedding.objects.get(pk=embedding_id)
    task_object = embedding_object.task
    show_progress = ShowProgress(task_object.id, multiplier=1)
    show_progress.update_step('phraser')
    show_progress.update_view(0)

    # parse field data
    field_data = [ElasticSearcher().core.decode_field_data(field) for field in embedding_object.fields]
    # create itrerator for phraser
    text_processor = TextProcessor(sentences=True, remove_stop_words=True, tokenize=True)
    sentences = ElasticSearcher(query=json.loads(embedding_object.query), field_data=field_data, output='text', callback_progress=show_progress, text_processor=text_processor)

    # build phrase model
    phraser = Phraser(embedding_id)
    phraser.build(sentences)

    # Number of word2vec passes + one pass to vocabulary building
    num_passes = 5
    total_passes = num_passes + 1

    # update progress
    show_progress = ShowProgress(task_object.id, multiplier=total_passes)
    show_progress.update_step('word2vec')
    show_progress.update_view(0)

    # build new processor with phraser
    text_processor = TextProcessor(phraser=phraser, sentences=True, remove_stop_words=True, tokenize=True)

    # iterate again with built phrase model to include phrases in language model
    sentences = ElasticSearcher(query=json.loads(embedding_object.query), field_data=field_data, output='text', callback_progress=show_progress, text_processor=text_processor)
    model = word2vec.Word2Vec(sentences, min_count=embedding_object.min_freq, size=embedding_object.num_dimensions, workers=NUM_WORKERS, iter=int(num_passes))

    # Save models
    show_progress.update_step('saving')
    model_path = os.path.join(MODELS_DIR, 'embedding', 'embedding_'+str(embedding_id))
    phraser_path = os.path.join(MODELS_DIR, 'embedding', 'phraser_'+str(embedding_id))
    model.save(model_path)
    phraser.save(phraser_path)

    # save model locations
    embedding_object.location = json.dumps({'embedding': model_path, 'phraser': phraser_path})
    embedding_object.vocab_size = len(model.wv.vocab)
    embedding_object.save()

    # declare the job done
    show_progress.update_step('')
    show_progress.update_view(100.0)
    task_object.update_status(Task.STATUS_COMPLETED, set_time_completed=True)

    return True


@task(name="train_tagger")
def train_tagger(tagger_id):
    # retrieve tagger & task objects
    tagger_object = Tagger.objects.get(pk=tagger_id)
    task_object = tagger_object.task

    show_progress = ShowProgress(task_object.id, multiplier=1)
    show_progress.update_step('scrolling positives')
    show_progress.update_view(0)

    field_data = [ElasticSearcher().core.decode_field_data(field) for field in tagger_object.fields]
    field_path_list = [field['field_path'] for field in field_data]

    # add phraser here
    if tagger_object.embedding:
        phraser = Phraser(embedding_id=tagger_object.embedding.pk)
        phraser.load()
        text_processor = TextProcessor(phraser=phraser, remove_stop_words=True)
    else:
        text_processor = TextProcessor(remove_stop_words=True)

    positive_samples = ElasticSearcher(query=json.loads(tagger_object.query), 
                                       field_data=field_data,
                                       output='doc_with_id',
                                       callback_progress=show_progress,
                                       scroll_limit=int(tagger_object.maximum_sample_size),
                                       text_processor=text_processor
                                       )

    positive_samples = list(positive_samples)
    positive_ids = set([doc['_id'] for doc in positive_samples])

    show_progress.update_step('scrolling negatives')
    show_progress.update_view(0)

    negative_samples = ElasticSearcher(field_data=field_data,
                                       output='doc_with_id',
                                       callback_progress=show_progress,
                                       scroll_limit=int(tagger_object.maximum_sample_size)*int(tagger_object.negative_multiplier),
                                       ignore_ids=positive_ids,
                                       text_processor=text_processor
                                       )

    negative_samples = list(negative_samples)

    show_progress.update_step('training')
    show_progress.update_view(0)

    tagger = TextTagger(tagger_id, field_list=field_path_list, classifier=tagger_object.classifier, vectorizer=tagger_object.vectorizer)
    tagger.train(positive_samples, negative_samples)

    show_progress.update_step('saving')
    show_progress.update_view(0)

    tagger_path = os.path.join(MODELS_DIR, 'tagger', 'tagger_'+str(tagger_id))
    tagger.save(tagger_path)

    # save model locations
    tagger_object.location = json.dumps({'tagger': tagger_path})
    tagger_object.statistics = json.dumps(tagger.statistics)
    tagger_object.save()

    # declare the job done
    show_progress.update_step('')
    show_progress.update_view(100.0)
    task_object.update_status(Task.STATUS_COMPLETED, set_time_completed=True)
    return True

