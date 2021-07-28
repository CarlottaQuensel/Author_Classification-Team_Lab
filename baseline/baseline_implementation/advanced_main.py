# -*- coding: utf-8 -*-
# Author: Carlotta Quensel
from nltk import tokenize
from nltk.tokenize.regexp import WordPunctTokenizer
from baseline import MaxEnt
from evaluation import Evaluation
from document import Poem
import advanced_Features
import pickle

# TODO: Set to the file path of the current folder
path = ""
# TODO: new pickle or csv for data
raw_data = pickle.load(open(f'{path}tokenized_dictionary.pickle', 'rb'))


def build_dataset(raw_data: dict[str], train_split: float = 0.75, min_poems: int = 30, max_author: int = None):
    """Builds a train and test set from the given poems sorted by author, with the 
    set constrictions of poem/poet number by converting the raw text poems into Poem objects
    with a vocabulary consisting of all words in the train set

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
    # Only look at the entries of authors who wrote at least min_poems number poems
    authors = sorted([author for author in raw_data.keys() if len(raw_data[author]) >=
                     min_poems and author != "Anonymous"], reverse=True, key=lambda x: len(raw_data[x]))
    if max_author:
        authors = authors[:max_author]
    # Split the data into a training and testing part, the default is a 75-25 ratio
    train, test = list(), list()
    # Types are all the unique tokens in the data later used as the vocabulary
    types = set()
    tokenizer = WordPunctTokenizer()
    for author in authors:
        # Both train and test set should include poems from all chosen authors,
        # so the data is split for each author individually instead of globally
        round_split = int(train_split*len(raw_data[author]))
        for i, poem in enumerate(raw_data[author]):
            if i < round_split:
                # The words from the training data are used in the vocabulary
                types = types.union(set([token.lower() for token in tokenizer.tokenize(poem)]))
                # The train set consists of pairs of a poem and corresponding author label
                train.append((poem, author))
            else:
                # As the words in the test set will not be learned as features for the classifier, they 
                # are not added to the vocabulary
                test.append((poem, author))
    # The dictionary to relate the indices of word vectors to word types from the data set
    vocabulary = {word: index for index, word in enumerate(sorted(types))}
    # Converting the poems from strings into Poem objects with a word vector, verses and rhyme scheme as features
    train = [(Poem(poem, vocabulary), author) for poem, author in train]
    test = [(Poem(poem, vocabulary), author) for poem, author in test]

    return train, test

classifier = MaxEnt()
train_set, test_set = build_dataset(raw_data, max_author=30)

classifier.learnFeatures(train_set, class_features=30)
classifier.train(train_set, trace=True)

predicted = list()
gold = [doc[1] for doc in test_set]
for doc in test_set:
    predicted.append(classifier.classify(doc[0]))

eva = Evaluation(gold, predicted)
eva.fullEval()
