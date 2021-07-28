# -*- coding: utf-8 -*-
# Author: Katrin Schmidt
import pandas
import json


path = "/Users/katrin/Desktop/Master/Team_Lab/Author_Classification/Author_Classification-Team_Lab/data/"

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
            dictionary_of_poems[row[1]] = [str(row[4])]
        else:
            dictionary_of_poems[row[1]].append(str(row[4]))


json_object = json.dumps(dictionary_of_poems)
jsonFile = open("poems.json", "w")
jsonFile.write(json_object)
jsonFile.close()

#print(json_object) 