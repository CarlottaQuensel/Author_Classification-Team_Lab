import nltk
from nltk.tokenize import WordPunctTokenizer
import pandas



with open('/Users/katrin/Desktop/Master/Team Lab/Author_Classification-Team_Lab-1/data/poetryfoundation-dataset.csv', encoding="utf-8") as f:
    data = pandas.read_csv(f)
    dictionary_of_poems = data.set_index('Author')['Content'].to_dict()
    list_of_poems = []
    for Author in dictionary_of_poems:
        poem = dictionary_of_poems.get(Author)
        list_of_poems.append([poem]) # list_of_poems = []  #[[poem1], [poem2], [poem3], ...]
    #first2pairs = {k: dictionary_of_poems[k] for k in list(dictionary_of_poems)[:5]}
    print()

['string....']
{Wendy Videlock: [['poem...'],[poem2],[...] ....}
    #tokenize_punctuation = WordPunctTokenizer()
    #tokens = tokenize_punctuation.tokenize()
    #dictionary_of_poems = data.set_index('Author').T.to_dict('list')
        
