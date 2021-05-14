# # -*- coding: utf-8 -*-
# TODO: turn into class
# TODO: streamline sorting option

class Evaluation():
    """Evaluation for a trained mulit-class single-label classifier.

    Methods:
    recall - compute the recall for every or a specific label
    precision - compute the precision for every or a specific label
    f1_score - compute the balanced f-score (alpha=0.5) from recall and precision
    confusion_matrix - erroneous classification as contingency table between gold and predicted labels
    """

    def __init__(self, gold_output, predicted_output, labels):
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
            raise IndexError("Unequal amount of labelled instances for gold and predicted labels. Use set_predictions to set again.")
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
        
        self.confusionMatrix = self.setConfusionMatrix()
    
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
        if len(gold_output) != len(predicted_output):
            raise IndexError("Unequal amount of labelled instances for gold and predicted labels. Use set_predictions to set again.")
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
        self.setConfusionMatrix()

    
    def setConfusionMatrix(self, gold_output=None, predicted_output=None):
        """Generate confusion matrix as a contingency table between predicted and gold labels.

        Args:
            gold_output (list(list(int)), optional): 
                    List of the classifications for all texts in the form of 1 for the true label and 0 for all others. 
                    Defaults to None, then the set class property is used.
            predicted_output (list(list(float), optional): 
                    List of the classification probabilities for all texts, where the highest score is the predicted label. 
                    Defaults to None, then the class property is used.
        """
        if gold_output and predicted_output:
            self.set_predictions(gold_output, predicted_output)
        elif not self.gold:
            raise ValueError("The labels to compute a confusion matrix are missing, use set_predictions first")
        
        # initialize contingency table with 0 counts for all labels
        confusion = [[0 for i in range(len(self.labels))] for j in range(len(self.labels))]
        for i in range(len(self.gold)):
            gold = self.gold[i].index(1)
            pred = self.predicted[i].index(1)
            confusion[gold][pred] += 1
        self.confusionMatrix = confusion


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
            try:
                # Compute the precision as the ratio of correct classifications in all predicted instances
                return self.confusionMatrix[label_index][label_index] / sum([self.confusionMatrix[j][label_index] for j in range(len(self.confusionMatrix))])
            except ZeroDivisionError:
                # If no instances were assigned the label, all no predictions were wrong
                return 1
        else:
            # Compute accumulated and average precision over all classes
            # The true positives correspond to the diagonal of the confusion matrix (where gold and predicted labels match up)
            tp_accumulated = sum([self.confusionMatrix[i][i] for i in range(len(self.confusionMatrix))])
            # The false positives correspond to the columns in the confusion matrix excluding the true positive (where the classifier predicted
            # the label instead of another gold label)
            fp_accumulated = sum([self.confusionMatrix[i][j] for i in range(len(self.confusionMatrix)) for j in range(len(self.confusionMatrix)) if i != j])
            # List for all individual precision scores to eventually compute the average
            mean = list()
            # Compute precision for every class individually as
            for i in range(len(self.confusionMatrix)):
                # correctly assigned label
                tp = self.confusionMatrix[i][i]
                fp = sum([self.confusionMatrix[j][i] for j in range(len(self.confusionMatrix)) if j != i])
                # divided by all instances that were assigned the label
                mean.append(tp/(tp+fp))
            self.prec = mean
            # Show the precision for each label, the average and the accumulated precision
            # TODO: pretty format for individual precision
            print(f"Precision\nAccumulated: {tp_accumulated/(tp_accumulated+fp_accumulated)}\nAverage: {sum(mean)/len(mean)}")
            return tp_accumulated/(tp_accumulated+fp_accumulated)

    
    def recall(self, label=None):
        """Compute recall for one or all labels as the ratio of gold labels that are predicted

        Args:
            label (string, optional): If set, only the recall score for this label is computed. Defaults to None.
        """
        if label:
            # If the label is given, compute recall for this one class
            try:
                # Find the label in the list of labels from the evaluated data
                label_index = [list(self.labels.keys())[list(self.labels.values()).index(label)]]
            except ValueError:
                # The label does not correspond to any label in the data
                print(f"The given label {label} is not part of the label set")
                return None
            try:
                # Compute the recall as the ratio of gold instances of the label which were found by the classifier
                return self.confusionMatrix[label_index][label_index] / sum(self.confusionMatrix[label_index])
            except ZeroDivisionError:
                # If none of the gold instances were found by the classifier, the recall is 0
                return 0
        else:
            # Compute accumulated and average recall over all classes
            # The true positives correspond to the diagonal of the confusion matrix (where gold and predicted labels match up)
            tp_accumulated = sum([self.confusionMatrix[i][i] for i in range(len(self.confusionMatrix))])
            # The false negatives correspond to the rows in the confusion matrix excluding the true positive (where the gold label was
            # predicted as another label)
            fn_accumulated = sum([self.confusionMatrix[i][j] for i in range(len(self.confusionMatrix)) for j in range(len(self.confusionMatrix)) if i != j])
            # List for all individual recall scores to eventually compute the average
            mean = list()
            # Compute recall for every class individually as
            for i in range(len(self.confusionMatrix)):
                # correctly assigned label
                tp = self.confusionMatrix[i][i]
                fn = sum([self.confusionMatrix[i][j] for j in range(len(self.confusionMatrix)) if j != i])
                # divided by all instances that should be assigned the label
                mean.append( tp / (tp+fn) )
            self.rec = mean
            # Show the recall for each label, the average and the accumulated recall
            # TODO: pretty format for individual recall
            print(f"Recall\nAccumulated: {tp_accumulated/(tp_accumulated+fn_accumulated)}\nAverage: {sum(mean)/len(mean)}")
            return tp_accumulated/(tp_accumulated+fn_accumulated)
    

    def f_score(self, alpha=0.5):
        # TODO: finish f-score
        return None                       
