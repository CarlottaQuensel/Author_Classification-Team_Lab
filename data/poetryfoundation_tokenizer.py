import nltk
from nltk.tokenize import WordPunctTokenizer
import pandas
import pickle
# TODO: think about replacing '\n' by ' .' --> if ' .' is followed by '.'


# open csv file with pandas
with open('/Users/katrin/Desktop/Master/Team Lab/Author_Classification-Team_Lab-1/data/poetryfoundation-dataset.csv', encoding="utf-8") as file:
    data_frame = pandas.read_csv(file)

    # initiate an empty dictionary
    dictionary_of_poems = {}

    # iterate over rows in data_frame,
    # add 'Author' and 'Content' to the dict,
    # so that dictionary_of_poems = {'Author': ['poem1', 'poem2',...]}
    for index, row in data_frame.iterrows():
        if row[1] not in dictionary_of_poems:
            dictionary_of_poems[row[1]] = [str(row[4]).replace('\n', ' ')] #[str([row[4]]).strip('\n')]
        else:
            dictionary_of_poems[row[1]].append(str(row[4]).replace('\n', ' '))

    # create a smaller data_frame for testing
    # data_frame_head = {k: dictionary_of_poems[k] for k in list(dictionary_of_poems)[2:6]}

    # tokenizing the data with nltk
    tokenize_punctuation = WordPunctTokenizer()
    tokenized_dictionary = {}

    # iterate over Authors in dictionary
    for Author in dictionary_of_poems:

        # iterate over each poem per Author
        for poem in dictionary_of_poems[Author]:
            tokenized_poem = tokenize_punctuation.tokenize(poem)
            
            # create a new dictionary with tokenized poems
            if Author not in tokenized_dictionary:
                tokenized_dictionary[Author] = [tokenized_poem]
            else:
                tokenized_dictionary[Author].append(tokenized_poem)
    pickle.dump(tokenized_dictionary, open('tokenized_dictionary.pickle', 'wb'))
    #print(tokenized_dictionary)
