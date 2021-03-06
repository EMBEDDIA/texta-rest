import json
import tempfile
import zipfile
from typing import List, Optional

from django.contrib.auth.models import User
from django.core import serializers
from django.db import models, transaction
from texta_lexicon_matcher.lexicon_matcher import LexiconMatcher, SUPPORTED_MATCH_TYPES, SUPPORTED_OPERATORS

from toolkit.constants import MAX_DESC_LEN
from toolkit.core.project.models import Project
from toolkit.core.task.models import Task
from toolkit.elastic.core import ElasticCore
from toolkit.elastic.document import ElasticDocument
from toolkit.settings import TEXTA_TAGS_KEY


def load_matcher(regex_tagger_object):
    # parse lexicons
    lexicon = json.loads(regex_tagger_object.lexicon)
    counter_lexicon = json.loads(regex_tagger_object.counter_lexicon)
    # create matcher
    matcher = LexiconMatcher(
        lexicon,
        counter_lexicon=counter_lexicon,
        operator=regex_tagger_object.operator,
        match_type=regex_tagger_object.match_type,
        required_words=regex_tagger_object.required_words,
        phrase_slop=regex_tagger_object.phrase_slop,
        counter_slop=regex_tagger_object.counter_slop,
        n_allowed_edits=regex_tagger_object.n_allowed_edits,
        return_fuzzy_match=regex_tagger_object.return_fuzzy_match,
        ignore_case=regex_tagger_object.ignore_case,
        ignore_punctuation=regex_tagger_object.ignore_punctuation
    )
    return matcher


class RegexTagger(models.Model):
    MODEL_TYPE = "regex_tagger"
    MODEL_JSON_NAME = "model.json"

    description = models.CharField(max_length=MAX_DESC_LEN)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    lexicon = models.TextField(default='')
    counter_lexicon = models.TextField(default='')
    operator = models.CharField(max_length=25, default=SUPPORTED_OPERATORS[0])
    match_type = models.CharField(max_length=25, default=SUPPORTED_MATCH_TYPES[0])
    required_words = models.FloatField(default=1.0)
    phrase_slop = models.IntegerField(default=0)
    counter_slop = models.IntegerField(default=0)
    n_allowed_edits = models.IntegerField(default=0)
    return_fuzzy_match = models.BooleanField(default=True)
    ignore_case = models.BooleanField(default=True)
    ignore_punctuation = models.BooleanField(default=True)


    def __str__(self):
        return self.description


    def match_texts(self, texts: List[str], as_texta_facts: bool = False, field="", matcher: Optional[LexiconMatcher] = None):
        results = []
        matcher = matcher or load_matcher(self)  # Optionally, you can insert another matcher of your choice.
        for text in texts:
            if text and isinstance(text, str):
                matches = matcher.get_matches(text)
                if as_texta_facts:
                    new_facts = []
                    for match in matches:
                        source_string = json.dumps({"regextagger_id": self.pk})
                        new_fact = {
                            "fact": self.description,
                            "str_val": match["str_val"],
                            "spans": json.dumps([match["span"]]),
                            "doc_path": field,
                            "source": source_string
                        }
                        new_facts.append(new_fact)
                    results.extend(new_facts)
                else:
                    results.extend(matches)
        return results


    def get_description(self):
        return {"tagger_id": self.pk, "description": self.description}


    def export_resources(self):
        with tempfile.SpooledTemporaryFile(encoding="utf8") as tmp:
            with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as archive:
                # Write model object to zip as json
                model_json = serializers.serialize("json", [self]).encode("utf8")
                archive.writestr(self.MODEL_JSON_NAME, model_json)
            tmp.seek(0)
            return tmp.read()


    @staticmethod
    def import_resources(zip_file, request, pk) -> int:
        with transaction.atomic():
            with zipfile.ZipFile(zip_file, "r") as archive:
                json_string = archive.read(RegexTagger.MODEL_JSON_NAME).decode()
                model_json = json.loads(json_string)[0]["fields"]
                del model_json["project"]
                del model_json["author"]
                # create new object
                new_model = RegexTagger(**model_json)
                # update user & project
                new_model.author = User.objects.get(id=request.user.id)
                new_model.project = Project.objects.get(id=pk)
                new_model.save()
                return new_model.id


class RegexTaggerGroup(models.Model):
    MODEL_TYPE = "regex_tagger_group"
    MODEL_JSON_NAME = "model.json"

    description = models.CharField(max_length=MAX_DESC_LEN)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    regex_taggers = models.ManyToManyField(RegexTagger, default=None)

    task = models.OneToOneField(Task, on_delete=models.SET_NULL, null=True)


    def apply(self, texts: List[str] = [], field_path: Optional[str] = None):
        results = []
        for text in texts:
            if isinstance(text, str) and text:
                for tagger in self.regex_taggers.all():
                    matcher = load_matcher(tagger)
                    matches = matcher.get_matches(text)
                    if field_path:
                        texta_facts = [{"str_val": tagger.description, "spans": json.dumps([match["span"]]), "fact": self.description, "doc_path": field_path} for match in matches]
                    else:
                        texta_facts = [{"str_val": tagger.description, "spans": json.dumps([match["span"]]), "fact": self.description} for match in matches]
                    results.extend(texta_facts)
        return results


    def match_texts(self, texts: List[str], as_texta_facts: bool = False, field: str = ""):
        results = []
        for tagger in self.regex_taggers.all():
            matcher = load_matcher(tagger)
            for text in texts:
                if text and isinstance(text, str):
                    matches = matcher.get_matches(text)
                    if as_texta_facts:
                        new_facts = []

                        for match in matches:
                            source_string = json.dumps({"regextaggergroup_id": self.pk, "regextagger_id": tagger.pk})
                            new_fact = {
                                "fact": self.description,
                                "str_val": tagger.description,
                                "doc_path": field,
                                "spans": json.dumps([match["span"]]),
                                "source": source_string
                            }
                            new_facts.append(new_fact)

                    if as_texta_facts:
                        results.extend(new_facts)
                    else:
                        results.extend(matches)
        return results


    def tag_docs(self, fields: List[str], docs: List[dict]):
        # apply tagger
        for doc in docs:
            for field in fields:
                flattened_doc = ElasticCore(check_connection=False).flatten(doc)
                text = flattened_doc.get(field, None)
                matches_as_facts = self.match_texts([text], as_texta_facts=True, field=field)
                for fact in matches_as_facts:
                    fact.update(fact=self.description)

                pre_existing_facts = doc.get(TEXTA_TAGS_KEY, [])
                filtered_facts = ElasticDocument.remove_duplicate_facts(pre_existing_facts + matches_as_facts)
                doc[TEXTA_TAGS_KEY] = filtered_facts

        return docs
