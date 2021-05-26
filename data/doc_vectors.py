import pickle

token_dict = pickle.load(open('C:/Users/HP Envy/Documents/Uni/Master/SS21/topics in emotion analysis/Author_Classification-Team_Lab/data/tokenized_dictionary.pickle', 'rb'))

words = set()
used_authors = list()
for author in token_dict:
    if len(token_dict[author]) >= 30 and author != 'Anonymous':
        used_authors.append(author)
        for poem in token_dict[author]:
            types = set(poem)
            words = words.union(types)

words = sorted(words)
used_authors = sorted(used_authors)