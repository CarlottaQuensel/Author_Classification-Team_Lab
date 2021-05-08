import pandas as pd
import numpy as np

# TODO: turn into class

'''
actual_output       can be true (=1) or false (=0)
predicted_output    can be true (=1) or false (=0) 
                    the highest probability (float) converted to int [0,1]
'''

file_path = '/Users/katrin/Desktop/Master/Team Lab/data_new.csv'

#read the csv file
df = pd.read_csv(file_path)

#to be continued
def normalize_output(actual_output, predicted_output):
    probability_dict = sorted({})
    highest_probability = probability_dict.keys()[0]


def compute_tp_tn_fn_fp(actual_output, predicted_output):
    '''
    True positive: actual = 1, predicted = 1
    True negative: actual = 0, predicted = 0
    False positive: actual = 1, predicted = 0
    False negative: actual = 0, predicted = 1
	
    Function sums up the overlap of actual and predicted output
    '''
    tp = sum((actual_output == 1) & (predicted_output == 1))
    tn = sum((actual_output == 0) & (predicted_output == 0))
    fn = sum((actual_output == 1) & (predicted_output == 0))
    fp = sum((actual_output == 0) & (predicted_output == 1))
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
