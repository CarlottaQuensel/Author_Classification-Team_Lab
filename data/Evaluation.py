# # -*- coding: utf-8 -*-
# TODO: turn into class

'''
gold_output       can be true (=1) or false (=0)
predicted_output    can be true (=1) or false (=0) 
                    the highest probability (float) converted to int [0,1]
'''

file_path = '/Users/katrin/Desktop/Master/Team Lab/data_new.csv'

#read the csv file
df = pd.read_csv(file_path)

#to be continued
# TODO: streamline sorting option
def normalize_output(actual_output, predicted_output):
    probability_dict = sorted({})
    highest_probability = probability_dict.keys()[0]

def compute_tp_tn_fn_fp(gold_output, predicted_output):
    '''
    True positive: gold = 1, predicted = 1
    True negative: gold = 0, predicted = 0
    False positive: gold = 1, predicted = 0
    False negative: gold = 0, predicted = 1
	
    Function sums up the overlap of actual and predicted output
    '''
    tp = sum((gold_output == 1) & (predicted_output == 1))
    tn = sum((gold_output == 0) & (predicted_output == 0))
    fn = sum((gold_output == 1) & (predicted_output == 0))
    fp = sum((gold_output == 0) & (predicted_output == 1))
    return tp, tn, fp, fn

tp, tn, fp, fn = compute_tp_tn_fn_fp(df.actual_output, df.predicted_output)

def compute_precision(tp, fp):
    '''
    Precision = tp  / tp + fp
    '''
    precision = tp/(tp + fp)
    return precision


def compute_recall(tp, fn):
    '''
    Recall = tp / tp + fn
    '''
    recall = tp/(tp + fn)
    return recall

precision = compute_precision(tp, fp)
recall = compute_recall(tp, fn)

def compute_f1_score(precision, recall):
    '''
    Calculates the F1 score
    '''
    f1_score = (2*precision*recall)/ (precision + recall)
    return f1_score



print(compute_precision(tp, fp))
print(compute_recall(tp, fn))


class evaluate():
    """Evaluation for a trained single-label classifier.

    Methods:
    recall - compute the recall for every or a specific label
    precision - compute the precision for every or a specific label
    f1_score - compute the balanced f-score (alpha=0.5) from recall and precision
    confusion_matrix - erroneous classification as contingency table between gold and predicted labels
    """

    def __init__(gold_output, predicted_output, labels):
        """[summary]

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
        
    def precision(label=None):
        """compute precision for one or all labels

        Args:
            label (string, optional): If set, only the precision score for this label is computed. Defaults to None.
        """
        if label:
            # find label in the label dictionary and compute precision for the respective class
            # else compute precision for all classes
            

            