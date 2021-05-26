import pickle

token_dict = pickle.load(open('C:/Users/HP Envy/Documents/Uni/Master/SS21/topics in emotion analysis/Author_Classification-Team_Lab/data/tokenized_dictionary.pickle', 'rb'))


words = set()
used_authors = list()
# TODO: convert into a pretty function
#       where the min number of poems or max number of authors can be scaled
for author in token_dict:
    if len(token_dict[author]) >= 30 and author != 'Anonymous':
        used_authors.append(author)
        for poem in token_dict[author]:
            types = set(poem)
            words = words.union(types)

# Use this list of words as a indexing guide for the word vectors
# --> 38814 types
words = sorted(words)
# 39 named authors with at least 30 poems
used_authors = sorted(used_authors)

# TODO: word vectors (lists of 0 and 1 or set of indexes with 1's)
#       words -> {I=1, think=2, but=3, am=4, a=5, cat=6, can=7, speak=8}
#       doc1 = ["I", "think"] -> [1,1,0,0,0,0,0,0] or {1,2}
#       doc2 = ["I", "am", "a", "cat"] -> [1,0,0,1,1,1,0,0] or {1,4,5,6}