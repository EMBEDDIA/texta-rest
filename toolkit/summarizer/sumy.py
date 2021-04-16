import os
from sumy.nlp.stemmers import null_stemmer
from sumy.parsers.plaintext import PlaintextParser

class SumyTokenizer:
    """
    Custom tokenizer for sumy.
    """
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

    def run_on_tokenized(self, text, summarizer_names):
        summarizers = self.get_summarizers(summarizer_names)

        stack = []
        parser = PlaintextParser.from_string(text, SumyTokenizer())

        summaries = {}
        for name, summarizer in summarizers.items():
            try:
                summarization = summarizer(parser.document, 1)
            except Exception as e:
                print(e)
                continue

            summary = [sent._text for sent in summarization]
            summary = "\n".join(summary)
            summaries[name] = summary

        stack.append(summaries)

        return stack

