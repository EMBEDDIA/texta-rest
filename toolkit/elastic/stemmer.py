from toolkit.elastic.core import ElasticCore
from elasticsearch.client import IndicesClient


class ElasticStemmer:

    def __init__(self):
        self.core = ElasticCore()
        self.indices_client = IndicesClient(self.core.es)
        
    
    def lemmatize(self, text):
        body = {
            "analyzer": "snowball",
            "text": text
        }

        # TODO: make analyze list of texts to make it faster
        # TODO: add exceptions etc.

        analysis = self.indices_client.analyze(body=body)

        tokens = [token["token"] for token in analysis["tokens"]]
        token_string = " ".join(tokens)

        return token_string
