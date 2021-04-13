LABEL_DISTRIBUTION = (
    ("random", "random"),
    ("original", "original"),
    ("equal", "equal"),
    ("custom", "custom")
)


# TODO Generate with ES6 and ES7 differences in mind.
def get_snowball_choices():
    elastic_langs = [
        'estonian', 'english', 'arabic', 'armenian', 'basque', 'bengali', 'brazilian',
        'bulgarian', 'catalan', 'czech', 'danish', 'dutch', 'finnish', 'french', 'galician',
        'german', 'greek', 'hindi', 'hungarian', 'indonesian', 'irish', 'italian', 'latvian',
        'lithuanian', 'norwegian', 'persian', 'portuguese', 'romanian', 'russian',
        'spanish', 'swedish', 'turkish', 'thai']

    choices = [(None, None)]

    for lang in elastic_langs:
        choices.append((lang, lang))

    return choices
