# TEXTA Toolkit 2

## Documentation

https://docs.texta.ee

## Wiki

https://git.texta.ee/texta/texta-rest/wikis/home

## Notes

Works with Python 3.6

Creating environment:

`conda env create -f environment.yaml`

Running migrations:

`python3 migrate.py`

Running application:

`python3 manage.py runserver`

`celery -A toolkit.taskman worker -l info`

Import testing data:

`python3 import_test_data.py`

Run all tests:

`python3 manage.py test`

Run tests for specific app:

`python3 manage.py test appname (eg python3 manage.py test toolkit.neurotagger)`

Run performance tests (not run by default as they are slow):

`python3 manage.py test toolkit.performance_tests`

Building Docker:

`docker build -t texta-rest:latest -f docker/Dockerfile .`

Running Docker:

`docker run -p 8000:8000 texta-rest:latest`

Building Docker with GPU support:

`docker build -t texta-rest:gpu-latest -f docker/gpu.Dockerfile .`

Running Docker with GPU support requires NVIDIA Container Toolkit to be installed on the host machine: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker. When Container Toolkit is installed:

`docker run --gpus all -p 8000:8000 texta-rest:latest-gpu`

# Environment variables

## Deploy & Testing variables

* TEXTA_DEPLOY_KEY - Used to separate different Toolkit instances for cases where Elasticsearch or the database are
  shared amongst multiple instances. Best to give this a simple number (Default: 1).
* TEXTA_ADMIN_PASSWORD - Password of the admin user created on first run.
* TEXTA_USE_CSRF - Disable CSRF for integration tests.
* TEXTA_CELERY_ALWAYS_EAGER - Whether to use Celerys async features or not, useful for testing purposes locally. (Default: False)
* TEXTA_SHORT_TASK_WORKERS - Number of processes available for short term tasks (Default: 3).
* TEXTA_LONG_TASK_WORKERS - Number of processes available for long term tasks (Default: 5).
* TEXTA_MLP_TASK_WORKERS - Number of processes available for MLP based tasks (Default: 2).
* TEXTA_RELATIVE_MODELS_DIR - Relative path of the directory in which all the different types of models are stored in.
  (Default: "/data/models").
* TEXTA_LANGUAGE_CODES - Comma separated string of Stanza supported language codes to use for Multilingual Processing.
  (Default: "et,en,ru").
* TEXTA_MLP_MODEL_DIRECTORY_PATH - Relative path to the directory into which Stanza models will be stored under the "
  stanza" folder (setting this to ./home/texta will create ./home/texta/stanza which contains subfolders for every
  language like ./home/texta/stanza/et etc). (Default: "./data/models").
* TEXTA_ALLOW_BERT_MODEL_DOWNLOADS - Boolean flag indicating if the users can download additional BERT models.
  (Default: False).
* TEXTA_BERT_MODEL_DIRECTORY_PATH - Relative path to the directory into which pretrained and fine-tuned BERT models will
  be stored under the "bert_tagger" folder. (setting this to ./home/texta will create ./home/texta/bert_tagger/pretrained/
  which contains subfolders for every downloaded bert more like ./home/texta/bert_model/pretrained/bert-base-multilingual-cased
  etc and ./home/texta/bert_model/fine_tuned/ which will store fine-tuned BERT models. (Default: "./data/models").
* TEXTA_BERT_MODELS - Comma seprated string of pretrained BERT models to download.
  (Default: "bert-base-multilingual-cased,bert-base-uncased,EMBEDDIA/finest-bert").
* SKIP_BERT_RESOURCES - If set "True", skips downloading pretrained BERT models. (Default: False).
* TEXTA_EVALUATOR_MEMORY_BUFFER_GB - The minimum amount of memory that should be left free while using the evaluator, unit = GB. (Default = 5GB)

## External services

* TEXTA_ES_URL - URL of the Elasticsearch instance including the protocol, host and port (ex. http://localhost:9200).
* TEXTA_REDIS_URL - URL of the Redis instance including the protocol, host and port (ex. redis://localhost:6379).

## Django specifics

* TEXTA_ES_PREFIX - String used to limit Elasticsearch index access. Only indices matched will be the ones matching "
  {TEXTA_ES_PREFIX}*".
* TEXTA_CORS_ORIGIN_WHITELIST - Comma separated string of urls (**NO WHITESPACE**) for the CORS whitelist. Needs to
  include the protocol (ex. http://* or http://*,http://localhost:4200).
* TEXTA_ALLOWED_HOSTS - Comma separated string (**NO WHITESPACE**) representing the host/domain names that this Django
  site can serve (ex. * or *,http://localhost:4200).
* TEXTA_DEBUG - True/False values on whether to run Django in it's Debug mode or not.
* TEXTA_SECRET_KEY - String key for cryptographic security purposes. ALWAYS SET IN PRODUCTION.

## Database credentials

* DJANGO_DATABASE_ENGINE - https://docs.djangoproject.com/en/3.0/ref/settings/#engine
* DJANGO_DATABASE_NAME - The name of the database to use. For SQLite, it’s the full path to the database file. When
  specifying the path, always use forward slashes, even on Windows.
* DJANGO_DATABASE_USER - The username to use when connecting to the database. Not used with SQLite.
* DJANGO_DATABASE_PASSWORD - The password to use when connecting to the database. Not used with SQLite.
* DJANGO_DATABASE_HOST - Which host to use when connecting to the database. An empty string means localhost. Not used
  with SQLite.
* DJANGO_DATABASE_PORT - The port to use when connecting to the database. An empty string means the default port. Not
  used with SQLite.

## Extra Elasticsearch connection configurations

Unless you have a specially configured Elasticsearch instance, you can ignore these options.

* TEXTA_ES_USER - Username to authenticate to a secured Elasticsearch instance.
* TEXTA_ES_PASSWORD - Password to authenticate to a secured Elasticsearch instance.

https://elasticsearch-py.readthedocs.io/en/6.3.1/connection.html#elasticsearch.Urllib3HttpConnection:

* TEXTA_ES_USE_SSL
* TEXTA_ES_VERIFY_CERTS
* TEXTA_ES_CA_CERT_PATH
* TEXTA_ES_CLIENT_CERT_PATH
* TEXTA_ES_CLIENT_KEY_PATH
* TEXTA_ES_TIMEOUT
* TEXTA_ES_SNIFF_ON_START
* TEXTA_ES_SNIFF_ON_FAIL
