# # -*- coding: utf-8 -*-
# TODO: turn into class
# TODO: streamline sorting option

class evaluate():
    """Evaluation for a trained single-label classifier.

    Methods:
    recall - compute the recall for every or a specific label
    precision - compute the precision for every or a specific label
    f1_score - compute the balanced f-score (alpha=0.5) from recall and precision
    confusion_matrix - erroneous classification as contingency table between gold and predicted labels
    """

    def __init__(gold_output, predicted_output, labels):
        """Initialize evaluation with the gold labels and the predictions to be evaluated.

        Args:
            gold_output (list(list(int))): 
                    list of the gold classification labels used to evaluate the classifier
                    each classification consists of 1 for the gold label and 0 for all others
            predicted_output (list(list(float))): 
                    list of the labels predicted by the classifier
                    each classification consists of a list of probabilities for each label
            labels (list(str)):
                    label names for the classification probabilities
                    length must match the classifications of both output arguments
        """
        if len(gold_output) != len(predicted_output):
            raise IndexError("Unequal amount of gold and predicted labels. Use set_predictions to set again.")
            return self
        try:
            self.labels = {i: label for i, label in enumerate(labels)}
        except NameError:
            self.labels = {i: str(i) for i in range(len(gold_output[0]))}
        self.gold = gold_output
        # Save unaltered probabilities given by the classifier
        self.predicted_raw = predicted_output

        # For comparison with the gold labels, convert the probabilities into 1 for the most likely, and 0 for all other labels
        self.predicted = []
        for classification in self.predicted_raw:
            # Get index of highest probability score
            max_prediction = classification.index(max(classification))
            prediction = [0 for i in range(len(self.labels))]
            # Set the highest scoring label to 1
            prediction[max_prediction] = 1
            self.predicted.append(prediction)
    
    def set_predictions(self, gold_output, predicted_output, labels=None):
        """Set gold and predicted labels for new data.

        Args:
            gold_output (list(list(int))): 
                    list of the gold classification labels used to evaluate the classifier
                    each classification consists of 1 for the gold label and 0 for all others
            predicted_output (list(list(float))): 
                    list of the labels predicted by the classifier
                    each classification consists of a list of probabilities for each label
            labels (list(str)):
                    label names for the classification probabilities
                    length must match the classifications of both output arguments
        """
        try:
            self.labels = {i: label for i, label in enumerate(labels)}
        except TypeError:
            self.labels = {i: str(i) for i in range(len(gold_output[0]))}
        self.gold = gold_output
        # Save unaltered probabilities given by the classifier
        self.predicted_raw = predicted_output

        # For comparison with the gold labels, convert the probabilities into 1 for the most likely, and 0 for all other labels
        self.predicted = []
        for classification in self.predicted_raw:
            # Get index of highest probability score
            max_prediction = classification.index(max(classification))
            prediction = [0 for i in range(len(self.labels))]
            # Set the highest scoring label to 1
            prediction[max_prediction] = 1
            self.predicted.append(prediction)
        self.reset_errors()

    def reset_errors(self):
        """After swapping the predictions, the errors should be reset to 0.
        """
        self.errors = {label_index: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for label_index in self.labels}

        
    def count_errors(self):
        """Count correct and false labels as 
            TP - true positives are the correct labels
            TN - true negatives are correctly unlabelled instances
            FP - false positives are falsely labelled instances
            FN - false negatives are falsely not labelled
        """
        self.errors = {label_index: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for label_index in self.labels}
        for i in range(len(self.gold)):
            gold = self.gold.index(1)
            pred = self.predicted.index(1)
            if gold == pred:
                self.errors[gold]["tp"] += 1
            else:
                self.errors[gold]["fn"] += 1
                self.errors[pred]["fp"] += 1
         true_negs = set(self.labels).difference({gold,pred})
         for label in true_negs:
             self.errors[label]["tn"] += 1


    def precision(self, label=None):
        """Compute precision for one or all labels as the ratio of predicted labels that are true

        Args:
            label (string, optional): If set, only the precision score for this label is computed. Defaults to None.
        """
        if label:
            # find label in the label dictionary and compute precision for the respective class
            try:
                label_index = [list(self.labels.keys())[list(self.labels.values()).index(label)]]
            except ValueError:
                print(f"The given label {label} is not part of the label set")
                return None
            return self.errors[label_index]["tp"] / (self.errors[label_index]["tp"] + self.errors[label_index]["fp"])
        else:
            label_index = list(self.labels.keys())

        # compute accumulated precision over all classes
        tp_accumulated = 0
        fp_accumulated = 0
        mean = list()
        for label in self.errors:
            tp_accumulated += self.errors[label]["tp"]
            fp_accumulated += self.errors[label]["fp"]
            mean.append(self.errors[label]["tp"]/(self.errors[label]["tp"]+self.errors[label]["fp"]))
        print(f"Precision\nAccumulated: {tp_accumulated/(tp_accumulated+fp_accumulated)}\nAverage: {sum(mean)/len(mean)}")
        return tp_accumulated/(tp_accumulated+fp_accumulated)
                        
