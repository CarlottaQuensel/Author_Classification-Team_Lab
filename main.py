# -*- coding: utf-8 -*-
# Author: Carlotta Quensel
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from classifier import MaxEnt
from evaluation import Evaluation
from document import Poem
import re
import json

# TODO: Set to the file path of the current folder
path = ""
raw_data = json.load(open(f'{path}poems.json', 'r'))


def build_dataset(
    raw_data: dict[str], train_split: float = 0.75,
    min_poems: int = 30, max_author: int = None
):
    """Use the specified number of most prolific authors to convert the given
    poems into a data set of Poem objects by tokenizing and extracting words,
    rhyme scheme and verse count. Only the training data is used in building
    a vocabulary for extracting the words.

    Args:
        raw_data (dict[str]):
            The poems as lists of strings sorted by author
        train_split (float, optional):
            Ratio of training and testing data. Defaults to 0.75.
        min_poems (int, optional):
            Least number of poems per author to be included in the data.
            Defaults to 30.
        max_author (int, optional):
            Maximum number of authors overall.
            Defaults to None.
    """
    # Sort authors with at least min_number of poems by prolificacy
    authors = sorted([author for author in raw_data.keys() if len(raw_data[author]) >=
                     min_poems and author != "Anonymous"],
                     reverse=True, key=lambda x: len(raw_data[x]))
    if max_author:
        authors = authors[:max_author]
    # Split the data into a train and test set, the default is a 75-25 ratio
    train, test = list(), list()
    # Types are all the unique tokens in the data later used as the vocabulary
    types = set()
    # Use NLTK's method to split punctuation from tokens
    tokenizer = WordPunctTokenizer()
    # Filter out punctuation-only tokens
    punctuation = "[^0-9A-Za-z]"
    for author in authors:
        # Split train and test set for each author rather than globally to
        # ensure that every author occurs in both sets
        round_split = int(train_split*len(raw_data[author]))
        for i, poem in enumerate(raw_data[author]):
            if i < round_split:
                # Poems allotted to the train set are used for the vocabulary
                tokens = set([token.lower() for token in tokenizer.tokenize(
                    poem) if len(re.sub(punctuation, "", token))])
                types = types.union(tokens)
                # Save the unprocessed poem and author in the right set
                train.append((poem, author))
            else:
                # Poems in the test set don't influence the vocabulary
                test.append((poem, author))
    # Clear the vocabulary from frequent words and save it for future reference
    types = types.difference(set(stopwords.words('english')).union({""}))
    vocabulary = {word: index for index, word in enumerate(sorted(types))}
    # Extract rhyme scheme, verse count and vocabulary words from the poems
    train = [(Poem(poem, vocabulary), author) for poem, author in train]
    test = [(Poem(poem, vocabulary), author) for poem, author in test]
    # Return the converted poems split into training and test data and the
    # corresponding vocabulary
    return train, test, vocabulary


classifier = MaxEnt()
# Build trrain and test data for the 30 most prolific authors
train_set, test_set, vocabulary = build_dataset(raw_data, max_author=30)
print(
    f"The vocabulary consists of {len(vocabulary)} words.",
    f"\nTrain data: {len(train_set)} poems, test data: {len(train_set)} poems")

# Train the baseline classifier with just word features
classifier.learnFeatures(train_set, bow_features=30, verse_features=False,
                         rhyme_features=0, vocabulary=vocabulary, trace=True)
classifier.train(train_set, trace=True)

# Evaluate the classifier on the test data
predicted = list()
gold = [doc[1] for doc in test_set]
for doc in test_set:
    predicted.append(classifier.classify(doc[0]))
eva = Evaluation(gold, predicted)
eva.fullEval()
