from collections import OrderedDict
from time import time

class ModelCache:
    """
    Cache to hold recently used Tagger & Embedding objects in memory.
    """
    def __init__(self, object_class):
        self.models = {}
        self.object_class = object_class


    def get_model(self, model_id):
        # load model if not in cache
        if model_id not in self.models:
            model = self.object_class(model_id)
            model.load()
            self.models[model_id] = {'model': model, 'last_access': time()}
        
        # update last access timestamp & remove old models
        self.models[model_id]['last_access'] = time()
        self.clean_cache()

        # return model
        return self.models[model_id]['model']
    

    def clean_cache(self):
        # removes models not accessed in last 60 minutes
        self.models = {k:v for k,v in self.models.items() if v['last_access'] >= time()-3600}
