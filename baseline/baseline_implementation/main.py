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

def count_verses(self):
    # open csv file with pandas
    with open('/Users/katrin/Desktop/Master/Team_Lab/Author_Classification/Author_Classification-Team_Lab/data/poetryfoundation-dataset.csv', encoding="utf-8") as file:
        data_frame = pandas.read_csv(file)
    
    # initiate an empty dictionary
    dictionary_of_poems = {}

    # iterate over rows in data_frame,
    # add 'Author' and 'Content' to the dict,
    # so that dictionary_of_poems = {'Author': [['poem1'], ['poem2'],[...], ...]}
    # where every poem = ['verse_1', 'verse_2', 'verse_3',...]
    for index, row in data_frame.iterrows():
        if row[1] not in dictionary_of_poems:
            dictionary_of_poems[row[1]] = [str(row[4]).splitlines()]
        else:
            dictionary_of_poems[row[1]].append(str(row[4]).splitlines())

    # iterate over authors in dictionary
    # and count verses,  create new dictionary
    # so that {Author: [count of verse1, count of verse2, ...], ...}
    new_dictionary = {}
    for Author in dictionary_of_poems:
        for poem in dictionary_of_poems[Author]:
            count_verses = (len(poem))
            if Author not in new_dictionary:
                new_dictionary[Author] = [count_verses]
            else:
                new_dictionary[Author].append(count_verses)

    # compute average count of verses per author
    # so that {Author: [average count of verses], ...}
    average_verse_count = {}
    for Author in new_dictionary:
        for poem in new_dictionary[Author]:
            average_verses = sum([poem]) / len([poem])
            average_verse_count[Author] = average_verses

    #pickle.dump(average_dictionary, open('average_dictionary.pickle', 'wb'))

def set_bins(self, average_verse_count: dict[str[int]]):

    '''Method that assigns Authors to different bins,
    according to their average verse length.'''

    # Set bins
    list_smaller_5 = []
    list_smaller_10 = []
    list_smaller_25 = []
    list_smaller_50 = []
    list_smaller_75 = []
    list_smaller_100 = []
    list_smaller_150 = []
    list_smaller_200 = []
    list_bigger_200 = []

    all_lists = [list_smaller_5, list_smaller_10, list_smaller_25, list_smaller_50, list_smaller_75, list_smaller_100, list_smaller_150, list_smaller_200, list_bigger_200]
    all_keys = [range(0,5), range(6,10), range(11,25), range(26,50), range(51,75), range(76,100), range(101,150), range(151,200), range(200,3500)]
    
    # Iterate over authors in the dictionary 
    # and their average verse length, in order
    # to assign them to the bins.
    # so that list_smaller_X = [Author1, Author2, ...]
    for Author in average_verse_count:
        for verse_length in average_verse_count[Author]:
            if verse_length <= 5:
                list_smaller_5.extend(Author)
            elif verse_length <= 10 and verse_length > 5:
                list_smaller_10.extend(Author)
            elif verse_length <= 25 and verse_length > 10: 
                list_smaller_25.extend(Author)
            elif verse_length <= 50 and verse_length > 25:
                list_smaller_50.extend(Author)
            elif verse_length <= 75 and verse_length > 50:
                list_smaller_75.extend(Author)
            elif verse_length <= 100 and verse_length > 75:
                list_smaller_100.extend(Author)
            elif verse_length <= 150 and verse_length > 100:
                list_smaller_150.extend(Author)
            elif verse_length <= 200 and verse_length > 150:
                list_smaller_200.extend(Author)
            elif verse_length > 200:
                list_bigger_200.extend(Author)

    # returns features like
    # {(0,5): [author1, author2], (6,10): [author3, author4],...}
    verse_feature_dictionary = {}
    for certain_list in all_lists:
        for key in all_keys:
            verse_feature_dictionary[key] = [certain_list]
        
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
