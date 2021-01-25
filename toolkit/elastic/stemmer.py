from toolkit.elastic.core import ElasticCore
from elasticsearch.client import IndicesClient
from .exceptions import ElasticSnowballException


class ElasticStemmer:

    def __init__(self, language="english"):
        self.core = ElasticCore()
        self.indices_client = IndicesClient(self.core.es)
        self.snowball_filter = {"type": "snowball", "language": language}

    def lemmatize(self, text):
        body = {
            "tokenizer": "whitespace",
            "text": text,
            "filter": [self.snowball_filter]
            
        }
        try:
            analysis = self.indices_client.analyze(body=body)
        except:
            raise ElasticSnowballException("Snowball failed. Check Connection & payload!")

        tokens = [token["token"] for token in analysis["tokens"]]
        token_string = " ".join(tokens)

        return token_string
