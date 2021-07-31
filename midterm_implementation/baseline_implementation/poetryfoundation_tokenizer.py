# -*- coding: utf-8 -*-
# Author: Katrin Schmidt
from nltk.tokenize import WordPunctTokenizer
import pandas
import pickle

# TODO: Set to the file path of the current folder
path = ""


# ----------------------------------------------------
# --------------------LOADING DATA--------------------
# ----------------------------------------------------
# open csv file with pandas
with open(f'{path}poetryfoundation-dataset.csv', encoding="utf-8") as file:
    data_frame = pandas.read_csv(file)

    # initiate an empty dictionary
    dictionary_of_poems = {}

    # iterate over rows in data_frame,
    # add 'Author' and 'Content' to the dict,
    # so that dictionary_of_poems = {'Author': ['poem1', 'poem2',...]}
    for index, row in data_frame.iterrows():
        if row[1] not in dictionary_of_poems:
            dictionary_of_poems[row[1]] = [str(row[4]).replace('\n', ' ')]
        else:
            dictionary_of_poems[row[1]].append(str(row[4]).replace('\n', ' '))


# ----------------------------------------------------
# --------------------TOKENIZATION--------------------
# ----------------------------------------------------
# tokenizing the data with nltk
tokenize_punctuation = WordPunctTokenizer()
tokenized_dictionary = {}

# iterate over authors in dictionary
for author in dictionary_of_poems:

    # iterate over each poem per author
    for poem in dictionary_of_poems[author]:
        tokenized_poem = [token.lower() for token in tokenize_punctuation.tokenize(poem)]

        # create a new dictionary with tokenized poems
        if author not in tokenized_dictionary:
            tokenized_dictionary[author] = [tokenized_poem]
        else:
            tokenized_dictionary[author].append(tokenized_poem)
pickle.dump(tokenized_dictionary, open('tokenized_dictionary.pickle', 'wb'))


# 39 named authors with at least 30 poems (-> 1569 poems):
#       'Alfred, Lord Tennyson', 'Algernon Charles Swinburne', 'Alice Notley', 'Ben Jonson', 'Billy Collins', 'Carl Sandburg',
#      'Christina Rossetti', 'David Ferry', 'Dean Young', 'Edgar Lee Masters', 'Edmund Spenser', 'Emily Dickinson',
#       'Frank Stanford', 'George Herbert', 'Gwendolyn Brooks', 'Henry Wadsworth Longfellow', 'Jane Hirshfield', 'John Ashbery',
#      'John Donne', 'John Keats', 'John Milton', 'Kahlil Gibran', 'Kay Ryan', 'Percy sshe Shelley', 'Rae Armantrout',
#       'Robert Browning', 'Robert Herrick', 'Samuel Menashe', 'Samuel Taylor Coleridge', 'Sir Philip Sidney', 'Thomas Hardy',
#      'W. S. Di Piero', 'W. S. Merwin', 'Walt Whitman', 'William Blake', 'William Butler Yeats', 'William Shakespeare',
#       'William Wordsworth', 'Yusef Komunyakaa'
