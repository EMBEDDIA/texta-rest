import math

# Sklearn average functions
AVG_FUNCTIONS = ["binary", "micro", "macro", "samples", "weighted"]
AVG_CHOICES = [(c, c) for c in AVG_FUNCTIONS]

# Default average function for multilabel/multiclass
DEFAULT_AVG_FUNCTION = "macro"

# Max number of fact values retrieved with facts aggregation
DEFAULT_MAX_AGGREGATION_SIZE = 30000

# Max number of classes to caclulate the confusion matrix
# If the number of classes exceeds the allowed limit,
# an empty matrix is returned
DEFAULT_MAX_CONFUSION_CLASSES = 10


# Default min and max count of label to display its result
# in `filtered_average` and `individual_results`.
DEFAULT_MIN_COUNT = 1
DEFAULT_MAX_COUNT = math.inf


# Fields that can be used for ordering the results in
# `filtered_average` and `individual_results`
ORDERING_FIELDS = ["count", "precision", "recall", "f1_score", "accuracy"]
ORDERING_FIELDS_CHOICES = [(c, c) for c in ORDERING_FIELDS]
DEFAULT_ORDER_BY_FIELD = "count"

# Order results in descending order?
DEFAULT_ORDER_DESC = False

# Metrics used
METRICS = ["precision", "recall", "f1_score", "accuracy"]

# Available keys for setting metric restrictions
METRIC_RESTRICTION_FIELDS = ["max_score", "min_score"]

DEFAULT_SCROLL_SIZE = 500
DEFAULT_ES_TIMEOUT = 10
