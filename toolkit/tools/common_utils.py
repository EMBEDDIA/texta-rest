from itertools import zip_longest
from toolkit.settings import UPLOAD_PATH
import uuid
import os

def write_file_to_disk(file_object):
    """
    Writes file object to disk by creating a unique filename to avoid name conflicts.
    """
    file_name = f"file_{uuid.uuid4().hex}_{file_object.name}"
    file_path = os.path.join(UPLOAD_PATH, file_name)
    with open(file_path, "wb") as fh:
        fh.write(file_object.file.read())
    return file_path


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def grouper(n, iterable, fillvalue=None):
    """
    Iterating trough an iterator/generator with chunks
    of size n.
    """
    container = []

    args = [iter(iterable)] * n
    chunks = zip_longest(fillvalue=fillvalue, *args)
    for chunk in chunks:
        chunk = [chunk for chunk in chunk if chunk is not None]
        container.append(chunk)

    return container[0]


def format_tagger_prediction(tag: str, probability: float, tagger_id: int = None, ner_match: bool = False,
                      lexicon_id: int = None, result: bool = True) -> dict:
    """ Formats Tagger prediction to ensure the same set of keys from different output sources.
    """
    prediction = {
        "tag": tag,
        "probability": probability,
        "tagger_id": tagger_id,
        "ner_match": ner_match,
        "lexicon_id": lexicon_id,
        "result": result
    }
    return prediction
    

class DisableCSRFMiddleware(object):

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        setattr(request, '_dont_enforce_csrf_checks', True)
        response = self.get_response(request)
        return response
