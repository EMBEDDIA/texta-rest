from toolkit.elastic.core import ElasticCore
from elasticsearch.client import IndicesClient


class ElasticStemmer:

    def __init__(self, language="English"):
        self.core = ElasticCore()
        self.indices_client = IndicesClient(self.core.es)
        self.language = language
    
    def lemmatize(self, text):
        body = {
            "tokenizer": "whitespace",
            "text": text,
            "filter": [
                {
                    "type": "snowball",
                    "language": self.language
                }
            ]
            
        }

        # TODO: make analyze list of texts to make it faster
        # TODO: add exceptions etc.
        # TODO: check stop word removal

        analysis = self.indices_client.analyze(body=body)

        tokens = [token["token"] for token in analysis["tokens"]]
        token_string = " ".join(tokens)

        return token_string
