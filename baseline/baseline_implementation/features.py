# -*- coding: utf-8 -*-
# Author: Carlotta Quensel
from numpy import log


class Feature():
    label = str()
    property = int()

    def __init__(self, label: str, doc_property: int) -> None:
        """Initialization of a maximum entropy feature.
        A MaxEnt feature is a binary function that where a document either has
        a property related to a label or not.

        Args:
            label (str): The label whose weight the function switches on or off.
            doc_property (int): The index of the feature's word/property in the document vector.
        """
        self.label = label
        self.property = doc_property

    def apply(self, current_label: str, doc: list[int]) -> int:
        """Compares a maximum entropy feature to a current document and label,
        to decide if the MaxEnt classifier will apply the feature and thus
        include a weight for this instance (1) or not (0).

        Args:
            current_label (str): The label whose probability the classifier should compute.
            doc (list[int]): The document instance currently being classified,
            as a bag-of-words feature vector.

        Returns:
            1:  If the feature can be applied to the document and the label
                matches with the probability currently computed.
            0:  If the document doesn't match the property the feature needs
                or another label is currently considered.
        """
        # The function calls for the label to match the function and
        # for the document to include a word or property.
        switch = (self.label == current_label and doc[self.property])
        # The boolean on/off-switch is converted to a number
        # to be multiplied with the weight
        return int(switch)


def learnFeatures(data: list[tuple[tuple[int], str]], class_features: int = 50, vocab=None) -> dict[dict[int]]:
    """Learns the most informative features (= words) for a label set from the
    set of respective documents. The function uses mutual pointwise information
    to compute the most relevant features for each label.
    Features can be the occurence or absence of a word in the document.

    Args:
        data (list[int]): The set of documents given as word vectors consisting of 1's and 0's.
        class_features (int, optional): The number of features learned for each class (author). Defaults to 50.
        vocab (list[str], optional): A list of words that can be used to show which words are learned to make the 
            word vectors more explainable by showing the vocabulary

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
        descending_scores = sorted([property for property in pmi[label]], reverse=True, key=lambda x: pmi[label][x])
        # Then reassign the scores to the properties in the right order
        # Only return the number of features determined by the user
        class_features = min(len(descending_scores), class_features)
        for feature in descending_scores[:class_features]:
            # print(label, vocab[feature[0]], pmi[label][feature])
            # Save feature functions as their own class
            features.append(Feature(label, feature[0]))
    return features


def pointwiseMutualInformation(data: list[tuple[tuple[int], str]]) -> dict[dict[int]]:
    """Takes data points (document vectors) and their labels and computes
    the pointwise mutual information for each combination of feature and label.

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
    c_words = [sum([doc_vec[i] for (doc_vec, label) in data]) for i in range(vocabulary)]
    # The number of documents the words do not occur in, is the complement
    # given the overall document count
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

        # The frequency of label-word pairs can be computed as a vector sum
        # of document vectors which have the label
        try:
            c_words_labels[label] = [c_words_labels[label][j] + doc[j] for j in range(vocabulary)]
        except KeyError:
            c_words_labels[label] = doc

    # Same as the words alone, the joint probability with the labels is also
    # computed for documents not including the words
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

    # Compute pointwise mutual information as p(w,l)/(p(w)*p(l))
    # with w=word, l=label
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
