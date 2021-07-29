# -*- coding: utf-8 -*-
# Author: Carlotta Quensel
from nltk import tokenize
from nltk.tokenize.regexp import WordPunctTokenizer
from baseline import MaxEnt
from evaluation import Evaluation
from document import Poem
import advanced_features
import pickle
import pandas

# TODO: Set to the file path of the current folder
path = ""
token_data = pickle.load(open(f'{path}tokenized_dictionary.pickle', 'rb'))


def build_dataset(raw_data: dict[str], train_split: float = 0.75, min_poems: int = 30, max_author: int = None):
    """Reads a dictionary of tokenized documents sorted by author, chooses the
    data according to the set constrictions on the least number of poems per author
    or maximum number of poets overall and then converts the chosen data into a
    training and test set made of bag-of-word vectors with author labels and an
    accompanying vocabulary dictionary.

    Args:
        raw_data (dict[str]):
            The documents as lists of tokens sorted into lists by author
        train_split (float, optional):
            Ratio of training and testing data. Defaults to 0.8.
        min_poems (int, optional):
            Least number of poems per author to be included in the data.
            Defaults to 30.
        max_author (int, optional):
            Maximum number of authors overall.
            Defaults to None.
    """
    # Only look at the entries of authors who wrote at least min_poems number poems
    authors = sorted([author for author in raw_data.keys() if len(raw_data[author]) >= min_poems and author != "Anonymous"], reverse=True, key=lambda x: len(raw_data[x]))
    if max_author:
        authors = authors[:max_author]
    # Split the data into a training and testing part, the default is a 80-20 ratio
    train, test = list(), list()
    # Types are all the unique tokens in the data later used as the vocabulary
    types = set()
    for author in authors:
        # Both train and test set should include poems from all chosen authors,
        # so the data is split for each author individually instead of globally
        round_split = int(train_split*len(raw_data[author]))
        for i, poem in enumerate(raw_data[author]):
            poem = [token.lower() for token in poem]
            if i < round_split:
                # The words from the training data are used in the vocabulary
                types = types.union(set(poem))
                # The train set consists of pairs of a poem and corresponding author label
                train.append((poem, author))
            else:
                # The test set is build the same as the train set but its words
                # are not taken into account for the vocabulary as they will not
                # be used in training anyway
                test.append((poem, author))
    # The dictionary to relate the indices of word vectors to word types from the data set
    vocabulary = {word: index for index, word in enumerate(sorted(types))}
    # Converting the training data from token lists into word vectors
    train = tok_to_vec(train, vocabulary)
    test = tok_to_vec(test, vocabulary)
    
    return train, test


def tok_to_vec(data: list[tuple[list[str], str]], vocabulary: dict[str]) -> list[tuple[list[int], str]]:
    """Takes a list of poem-author pairs where the poems are saved as token lists
    and converts them into word vectors with the help of an external vocabulary.

    Args:
        data (list[tuple[list[str], str]]):
            List of the poem-author pairs where the poem representation is to be changed
        vocabulary (dict[str]):
            External dictionary that assigns indices in the word vector to each of its entries

    Returns:
        list[tuple[list[int], str]]: The same data list but with the poems changed
        from token lists to word vectors
    """
    # Each vector has the size of the vocabulary and consists of binary indicators
    # on the inclusion or absence of a word in the current poem
    vocab_size = len(vocabulary)
    # For every poem in tokenized form, a blank word vector is initialized
    # (with only 0's/no words)
    for i, (tok_poem, label) in enumerate(data):
        vec_poem = [0 for i in range(vocab_size)]
        # Every word included in the poem that is part of the vocabulary is marked
        # in the vector representation
        for token in tok_poem:
            try:
                vec_poem[vocabulary[token]] = 1
            except KeyError:
                # If the poem was not used in building the vocabulary, it might
                # include unknown words that are ignored
                continue
        # The vector representation replaces the token representation in the
        # poem-author pair
        data[i] = (vec_poem, label)
    # The list of converted datapoints is returned
    return data


        
classifier = MaxEnt()
train_set, test_set = build_dataset(token_data, max_author=30)


classifier.learnFeatures(train_set, class_features=30)
classifier.train(train_set, trace=True)

predicted = list()
gold = [doc[1] for doc in test_set]
for doc in test_set:
    predicted.append(classifier.classify(doc[0]))

eva = Evaluation(gold, predicted)
eva.fullEval()
