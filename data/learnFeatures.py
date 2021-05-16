# -*- coding: utf-8 -*-
# TODO: write function to learn the most informative features from a dataset
# TODO: what are possible MPI scores (need log(p(xy)) or just p(xy)?)

"""
Function (list of labelled docs)
    sum up/collapse document feature vectors:
    1. all docs: c(words)
    2. count in groups by label c(words,labels)
    3. count labels (no vector summation) for c(labels)
    4. take complement of c(words) and c(words,labels) for negative feature weights
    
    5. normalize 1 to 4 with the number of documents (length of doc list) -- c(x)->p(x)
    
    6. compute MPI log(  p(w,l) / (p(w)p(l)))  ) for all word/label combinations
    7. sort MPI scores first by label then descending
    
    8. return list of best feature tuples (label, word, included)
        a - best # per class
        b - scores above x (0.7?) but at least as in a
"""