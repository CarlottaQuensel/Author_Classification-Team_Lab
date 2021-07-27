# -*- coding: utf-8 -*-
#from baseline.baseline_implementation.main import *
import nltk
from nltk.tokenize import WordPunctTokenizer
import pandas
import pickle


 

class MetricFeatures():

    average_dictionary = {}

    def open_file(self):
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
        # and count verses
        new_dictionary = {}
        for Author in dictionary_of_poems:
            for poem in dictionary_of_poems[Author]:
                count_verses = (len(poem))
                if Author not in new_dictionary:
                    new_dictionary[Author] = [count_verses]
                else:
                    new_dictionary[Author].append(count_verses)

        # compute average count of verses per author
        average_dictionary = {}
        for Author in new_dictionary:
            for poem in new_dictionary[Author]:
                average_verses = sum([poem]) / len([poem])
                average_dictionary[Author] = average_verses

        pickle.dump(average_dictionary, open('average_dictionary.pickle', 'wb'))


    def set_bins(self):

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

        # Iterate over authors in the dictionary 
        # and their average verse length, in order
        # to assign them to the bins.
        for Author in self.average_dictionary:
            for verse_length in self.average_dictionary[Author]:
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


    # def determine_bin(self, poem):
    #     verse_length = len([poem.splitlines()])
    #     if verse_length <= 5:
    #         return self.list_smaller_5
    #     elif verse_length <= 10 and verse_length > 5:
    #         return self.list_smaller_10
    #     elif verse_length <= 25 and verse_length > 10: 
    #         return self.list_smaller_25
    #     elif verse_length <= 50 and verse_length > 25:
    #         return self.list_smaller_50     
    #     elif verse_length <= 75 and verse_length > 50:
    #         return self.list_smaller_75   
    #     elif verse_length <= 100 and verse_length > 75:
    #         return self.list_smaller_100
    #     elif verse_length <= 150 and verse_length > 100:
    #         return self.list_smaller_150
    #     elif verse_length <= 200 and verse_length > 150:
    #         return self.list_smaller_200   
    #     elif verse_length > 200:
    #         return self.list_bigger_200
                

metric_features = MetricFeatures()
metric_features.open_file()
metric_features.set_bins()
print(metric_features.open_file())

# 'And I think, that this is a good thing. \n As I know that it is. \n And now, bye.'
