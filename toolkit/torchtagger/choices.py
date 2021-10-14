from texta_torch_tagger.tagger import TORCH_MODELS

DEFAULT_MAX_SAMPLE_SIZE = 100000
DEFAULT_MIN_SAMPLE_SIZE = 50
DEFAULT_NEGATIVE_MULTIPLIER = 1
DEFAULT_NUM_EPOCHS = 5
DEFAULT_VALIDATION_SPLIT = 0.8

DEFAULT_REPORT_IGNORE_FIELDS = ["true_positive_rate", "false_positive_rate"]

MODEL_CHOICES = [(a, a) for a in TORCH_MODELS.keys()]

DEFAULT_BALANCE = False
DEFAULT_USE_SENTENCE_SHUFFLE = False
DEFAULT_BALANCE_TO_MAX_LIMIT = False
