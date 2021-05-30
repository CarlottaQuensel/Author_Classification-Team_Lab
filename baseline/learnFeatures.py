# -*- coding: utf-8 -*-
from features import Feature

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

def learnFeatures(data: list[tuple[tuple[int], str]], class_features: int = 50) -> dict[dict[int]]:
    """Learns the most informative features (= words) for a label set from the set of respective documents.
    The function uses mutual pointwise information to compute the most relevant features for each label.
    Features can be the occurence or absence of a word in the document.

    Args:
        data (list[int]): The set of documents given as word vectors consisting of 1's and 0's.
        labels (list[str]): The respective labels in the same order as the documents.

    Raises:
        IndexError: The labels belong to the documents, thus their number should match up

    Returns:
        dict[dict[int]]: document features sorted by label and in descending order by PMI score
    """
    
    pmi = pointwiseMutualInformation(data)
    # Sort PMI scores by relevance to find the most informative features
    features = list()
    for label in pmi:
        # First sort the document properties according to their PMI score
        descending_scores = sorted([property for property in pmi[label]], reverse=True, key = lambda x: pmi[label][x])
        # Then reassign the scores to the properties in the right order
        # Only return the number of features determined by the user
        class_features = min(len(descending_scores), class_features)
        for feature in descending_scores[:class_features]:
            # Save feature functions as their own class
            features.append(Feature(label, feature))
    print(f"The classifier learned {len(features)} features")
    return features


def pointwiseMutualInformation(data: list[tuple[tuple[int], str]]) -> dict[dict[int]]:
    """Takes data points (document vectors) and their labels and computes the pointwise mutual information for 
    each combination of feature and label.

    Args:
        data (list[int]): Documents given as word vectors with 1/included and 0/absent words as elements
        labels (list[str]): The respective labels for the documents

    Returns:
        dict[dict[int]]: PMI scores sorted by label and feature
    """
    doc_number = len(data)
    vocabulary = len(data[0][0])

    # As the document vectors consist of 1 and 0, summing all documents yields the 
    # number of documents each word occurs in.
    c_words = [sum([document[0][i] for document in data]) for i in range(vocabulary)]
    # The number of documents the words do not occur in, is the complement given the overall document count
    c_nwords = [doc_number - wordcount for wordcount in c_words]
    

    # Counting the frequency of each label
    c_labels = dict()
    # and each label-word pair
    c_words_labels = dict()
    for i, (doc, label) in enumerate(data):
        # Handling the first and subsequent occurences of a label
        try:
            c_labels[label] += 1
        except KeyError:
            c_labels[label] = 1

        # The frequency of label-word pairs can be computed as a vector sum of document vectors which have the label
        try:
            c_words_labels[label] = [c_words_labels[label][j] + doc[j] for j in range(vocabulary)]
        except KeyError:
            c_words_labels[label] = doc[:]
    
    # Same as the words alone, the joint probability with the labels is also computed for documents not including the words
    c_nwords_labels = dict()
    for label in c_words_labels:
        # These counts are the complement of the documents 
        c_nwords_labels[label] = [c_labels[label] - c_words_labels[label][i] for i in range(vocabulary)]

    # Convert frequencies/counts into relative frequencies/probabilities for all counts
    p_words = [c_words[i]/doc_number for i in range(len(c_words))]
    p_nwords = [c_nwords[i]/doc_number for i in range(len(c_nwords))]
    p_labels = {label: c_labels[label]/doc_number for label in c_labels}
    p_words_labels = {label: [c_words_labels[label][i]/doc_number for i in range(vocabulary)] for label in c_words_labels}
    p_nwords_labels = {label: [c_nwords_labels[label][i]/doc_number for i in range(vocabulary)] for label in c_nwords_labels}

    # Compute pointwise mutual information as p(w,l)/(p(w)*p(l)) with w=word, l=label
    pmi = dict()
    for label in p_labels:
        pmi[label] = dict()
        for i in range(vocabulary):
            # Scores for occurence of the word
            if c_words[i]:
                pmi[label][(i, True)] = p_words_labels[label][i] / (p_words[i] * p_labels[label])
            else:
                # Scores for absence of the word
                pmi[label][(i, False)] = p_nwords_labels[label][i] / (p_nwords[i] * p_labels[label])
    # Scores are sorted by label and feature (word) and returned
    return pmi