from sumy.nlp.stemmers import null_stemmer


class Sumy:

    def get_summarizers(names):
        summarizers = {}
        for name in names:
            if (name == "random"):
                from sumy.summarizers.random import RandomSummarizer
                summarizers["random"] = RandomSummarizer(null_stemmer)
            elif (name == "luhn"):
                from sumy.summarizers.luhn import LuhnSummarizer
                summarizers["luhn"] = LuhnSummarizer(stemmer=null_stemmer)
            elif (name == "lsa"):
                from sumy.summarizers.lsa import LsaSummarizer
                summarizers["lsa"] = LsaSummarizer(stemmer=null_stemmer)
            elif (name == "lexrank"):
                from sumy.summarizers.lex_rank import LexRankSummarizer
                summarizers["lexrank"] = LexRankSummarizer(null_stemmer)
            elif (name == "textrank"):
                from sumy.summarizers.text_rank import TextRankSummarizer
                summarizers["textrank"] = TextRankSummarizer(null_stemmer)
            elif (name == "sumbasic"):
                from sumy.summarizers.sum_basic import SumBasicSummarizer
                summarizers["sumbasic"] = SumBasicSummarizer(null_stemmer)
            elif (name == "kl-sum"):
                from sumy.summarizers.kl import KLSummarizer
                summarizers["kl-sum"] = KLSummarizer(null_stemmer)
            elif (name == "reduction"):
                from sumy.summarizers.reduction import ReductionSummarizer
                summarizers["reduction"] = ReductionSummarizer(null_stemmer)

        for _, summarizer in summarizers.items():
            summarizer.stop_words = frozenset(et_stopwords)

        return summarizers
