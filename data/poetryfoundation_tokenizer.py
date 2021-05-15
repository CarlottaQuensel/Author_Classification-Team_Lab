import nltk
from nltk.tokenize import WordPunctTokenizer
import pandas


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
            dictionary_of_poems[row[1]] = [row[4]]
        else:
            dictionary_of_poems[row[1]].append(row[4])

    # create a smaller data_frame for testing
    data_frame_head = {k: dictionary_of_poems[k] for k in list(dictionary_of_poems)[:5]}
    print(data_frame_head)


    #tokenize_punctuation = WordPunctTokenizer()
    #tokens = tokenize_punctuation.tokenize()
 