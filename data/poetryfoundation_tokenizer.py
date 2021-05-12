import nltk
from nltk.tokenize import WordPunctTokenizer
import pandas



with open('/Users/katrin/Desktop/Master/Team Lab/Author_Classification-Team_Lab-1/data/poetryfoundation-dataset.csv', encoding="utf-8") as f:
    data = pandas.read_csv(f)
    data_dict = data.set_index('Author').T.to_dict('list')
    first2pairs = {k: data_dict[k] for k in list(data_dict)[:2]}
    print(first2pairs)
    
    #tokenize_punctuation = WordPunctTokenizer()
    #tokens = tokenize_punctuation.tokenize()
        
