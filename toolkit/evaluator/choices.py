AVG_FUNCTIONS = ["binary", "micro", "macro", "samples", "weighted"]

AVG_CHOICES = [(c, c) for c in AVG_FUNCTIONS]
DEFAULT_AVG_FUNCTION = "macro"

# max number of fact values retrieved with facts aggregation
DEFAULT_MAX_AGGREGATION_SIZE = 10000

# Max number of classes to caclulate the confusion matrix
# If the number of classes exceed the allowed limit,
# an empty matrix is returned
DEFAULT_MAX_CONFUSION_CLASSES = 10
