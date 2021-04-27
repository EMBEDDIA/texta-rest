import os
from typing import List
from sumy.nlp.stemmers import null_stemmer
from sumy.parsers.plaintext import PlaintextParser
from pelecanus import PelicanJson

class SumyTokenizer:
    """
    Custom tokenizer for sumy.
    """

    @staticmethod
    def sentences_ratio(text, ratio):
        tkns = text.split("###")
        count = len(tkns)
        return count * ratio

    @staticmethod
    def to_sentences(text):
        return text.split("###")

    @staticmethod
    def to_words(sentence):
        return sentence.lower().split()


class Sumy:
    def get_stop_words(self):
        stop_words = {}
        stop_word_dir = os.path.join(os.path.dirname(__file__), 'stop_words')
        for f in os.listdir(stop_word_dir):
            with open('{0}/{1}'.format(stop_word_dir, f), encoding="utf8") as fh:
                for stop_word in fh.read().strip().split('\n'):
                    stop_words[stop_word] = True

        return stop_words

    def get_summarizers(self, names):
        summarizers = {}
        for name in names:
            if name == "random":
                from sumy.summarizers.random import RandomSummarizer
                summarizers["random"] = RandomSummarizer(null_stemmer)
            elif name == "luhn":
                from sumy.summarizers.luhn import LuhnSummarizer
                summarizers["luhn"] = LuhnSummarizer(stemmer=null_stemmer)
            elif name == "lsa":
                from sumy.summarizers.lsa import LsaSummarizer
                summarizers["lsa"] = LsaSummarizer(stemmer=null_stemmer)
            elif name == "lexrank":
                from sumy.summarizers.lex_rank import LexRankSummarizer
                summarizers["lexrank"] = LexRankSummarizer(null_stemmer)
            elif name == "textrank":
                from sumy.summarizers.text_rank import TextRankSummarizer
                summarizers["textrank"] = TextRankSummarizer(null_stemmer)
            elif name == "sumbasic":
                from sumy.summarizers.sum_basic import SumBasicSummarizer
                summarizers["sumbasic"] = SumBasicSummarizer(null_stemmer)
            elif name == "kl-sum":
                from sumy.summarizers.kl import KLSummarizer
                summarizers["kl-sum"] = KLSummarizer(null_stemmer)
            elif name == "reduction":
                from sumy.summarizers.reduction import ReductionSummarizer
                summarizers["reduction"] = ReductionSummarizer(null_stemmer)

        for _, summarizer in summarizers.items():
            summarizer.stop_words = frozenset(self.get_stop_words())

        return summarizers

    def parse_doc_texts(self, doc_path: str, document: dict) -> list:
        """
        Function for parsing text values from a nested dictionary given a field path.
        :param doc_path: Dot separated path of fields to the value we wish to parse.
        :param document: Document to be worked on.
        :return: List of text fields that will be processed by MLP.
        """
        wrapper = PelicanJson(document)
        doc_path_as_list = doc_path.split(".")
        content = wrapper.safe_get_nested_value(doc_path_as_list, default=[])
        if content and isinstance(content, str):
            return [content]
        # Check that content is non-empty list and there are only stings in the list.
        elif content and isinstance(content, list) and all([isinstance(list_content, str) for list_content in content]):
            return content
        # In case the field path is faulty and it gives you a dictionary instead.
        elif isinstance(content, dict):
            return []
        else:
            return []

    def run_on_tokenized(self, text, summarizer_names, ratio):
        summarizers = self.get_summarizers(summarizer_names)

        stack = []
        ratio_count = SumyTokenizer().sentences_ratio(text, float(ratio))
        parser = PlaintextParser.from_string(text, SumyTokenizer())

        summaries = {}
        for name, summarizer in summarizers.items():
            try:
                summarization = summarizer(parser.document, float(ratio_count))
            except Exception as e:
                print(e)
                continue

            summary = [sent._text for sent in summarization]
            summary = "\n".join(summary)
            summaries[name] = summary

        stack.append(summaries)

        return stack

    def run_on_index(self, docs: List[dict], doc_paths: List[str], ratio, algorithm=["lexrank"]):
        stack = []
        for document in docs:
            for doc_path in doc_paths:
                # Traverse the (possible) nested dicts and extract their text values from it as a list of strings.
                # Since the nested doc_path could lead to a list there are multiple pieces of text which would be needed to process.
                doc_texts = self.parse_doc_texts(doc_path, document)
                for raw_text in doc_texts:
                    summarizers = self.get_summarizers(algorithm)
                    ratio_count = SumyTokenizer().sentences_ratio(raw_text, float(ratio))
                    parser = PlaintextParser.from_string(raw_text, SumyTokenizer())

                    summaries = {}
                    for name, summarizer in summarizers.items():
                        try:
                            summarization = summarizer(parser.document, float(ratio_count))
                        except Exception as e:
                            print(e)
                            continue

                        summary = [sent._text for sent in summarization]
                        summary = "\n".join(summary)
                        summaries[name] = summary

            stack.append(summaries)

        return stack
