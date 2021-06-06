# -*- coding: utf-8 -*-
from features import Feature
from numpy import log


def learnFeatures(data: list[tuple[tuple[int], str]], class_features: int = 50, vocab=None) -> dict[dict[int]]:
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
            #print(label, vocab[feature[0]], pmi[label][feature])
            # Save feature functions as their own class
            features.append(Feature(label, feature[0]))
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
    c_words = [sum([doc_vec[i] for (doc_vec,label) in data]) for i in range(vocabulary)]
    # The number of documents the words do not occur in, is the complement given the overall document count
    c_nwords = [doc_number - wordcount for wordcount in c_words]
    

    # Counting the frequency of each label
    c_labels = dict()
    # and each label-word pair
    c_words_labels = dict()
    for doc, label in data:
        # Handling the first and subsequent occurences of a label
        try:
            c_labels[label] += 1
        except KeyError:
            c_labels[label] = 1

        # The frequency of label-word pairs can be computed as a vector sum of document vectors which have the label
        try:
            c_words_labels[label] = [c_words_labels[label][j] + doc[j] for j in range(vocabulary)]
        except KeyError:
            c_words_labels[label] = doc
    
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
            pmi[label][(i, True)] = log(p_words_labels[label][i] / (p_words[i] * p_labels[label]))
            # Scores for absence of the word
            pmi[label][(i, False)] = log(p_nwords_labels[label][i] / (p_nwords[i] * p_labels[label]))
    # The PMI scores are ordered first by label and then by document property and returned
    return pmi
