import math

AVG_FUNCTIONS = ["binary", "micro", "macro", "samples", "weighted"]

AVG_CHOICES = [(c, c) for c in AVG_FUNCTIONS]
DEFAULT_AVG_FUNCTION = "macro"

# max number of fact values retrieved with facts aggregation
DEFAULT_MAX_AGGREGATION_SIZE = 10000

# Max number of classes to caclulate the confusion matrix
# If the number of classes exceed the allowed limit,
# an empty matrix is returned
DEFAULT_MAX_CONFUSION_CLASSES = 10


DEFAULT_TAGS_TO_INCLUDE = []
DEFAULT_METRICS_TO_INCLUDE = ["precision", "recall", "f1_score", "accuracy", "confusion_matrix"]
METRIC_CHOICES = [(c, c) for c in DEFAULT_METRICS_TO_INCLUDE]


DEFAULT_MIN_COUNT = 1
DEFAULT_MAX_COUNT = math.inf #10000#math.inf

DEFAULT_ORDER_BY_FIELD = "count"

ORDERING_FIELDS = ["count", "precision", "recall", "f1_score", "accuracy"]
ORDERING_FIELDS_CHOICES = [(c, c) for c in ORDERING_FIELDS]
