from toolkit.elastic.core import ElasticCore
from elasticsearch.client import IndicesClient


class ElasticStemmer:

    def __init__(self):
        self.core = ElasticCore()
        self.indices_client = IndicesClient(self.core.es)
        
    
    def stem(self, text):
        body = {
            "analyzer": "snowball",
            "text": text
        }

        analysis = self.indices_client.analyze(body=body)

        tokens = [token["token"] for token in analysis["tokens"]]
        token_string = " ".join(tokens)

        return tokens

