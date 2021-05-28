import pickle

token_dict = pickle.load(open('C:/Users/HP Envy/Documents/Uni/Master/SS21/topics in emotion analysis/Author_Classification-Team_Lab/data/tokenized_dictionary.pickle', 'rb'))


vocabulary = set()
author_list = list()
# TODO: convert into a pretty function
#       where the min number of poems or max number of authors can be scaled
for author in token_dict:
    if len(token_dict[author]) >= 30 and author != 'Anonymous':
        author_list.append(author)
        for poem in token_dict[author]:
            types = set(poem)
            vocabulary = vocabulary.union(types)

# Use this list of words as a indexing guide for the word vectors
# --> 38814 word types
vocabulary = {word: i for i, word in enumerate(sorted(vocabulary))}

# 39 named authors with at least 30 poems -> 1576 poems
# 'Alfred, Lord Tennyson', 'Algernon Charles Swinburne', 'Alice Notley', 'Ben Jonson', 'Billy Collins', 
# 'Carl Sandburg', 'Christina Rossetti', 'David Ferry', 'Dean Young', 'Edgar Lee Masters', 'Edmund Spenser',
# 'Emily Dickinson', 'Frank Stanford', 'George Herbert', 'Gwendolyn Brooks', 'Henry Wadsworth Longfellow', 
# 'Jane Hirshfield', 'John Ashbery', 'John Donne', 'John Keats', 'John Milton', 'Kahlil Gibran', 'Kay Ryan', 
# 'Percy sshe Shelley', 'Rae Armantrout', 'Robert Browning', 'Robert Herrick', 'Samuel Menashe', 'Samuel Taylor Coleridge', 
# 'Sir Philip Sidney', 'Thomas Hardy', 'W. S. Di Piero', 'W. S. Merwin', 'Walt Whitman', 'William Blake', 
# 'William Butler Yeats', 'William Shakespeare', 'William Wordsworth', 'Yusef Komunyakaa'
author_list = sorted(author_list)


'''# word vectors (lists of 0 and 1 or set of indexes with 1's), e.g.:
#       words -> {I=1, think=2, but=3, am=4, a=5, cat=6, can=7, speak=8}
#       doc1 = ["I", "think"] -> [1,1,0,0,0,0,0,0] or {1,2}
#       doc2 = ["I", "am", "a", "cat"] -> [1,0,0,1,1,1,0,0] or {1,4,5,6}'''
data = set()
for author in author_list:
    for poem in token_dict[author]:
        vec = set()
        for token in poem:
            vec.add(vocabulary[token])
        data.add((tuple(vec), author))

pickle.dump(data, open('C:/Users/HP Envy/Documents/Uni/Master/SS21/topics in emotion analysis/Author_Classification-Team_Lab/data/data.pickle', 'wb'))
pickle.dump(vocabulary, open('C:/Users/HP Envy/Documents/Uni/Master/SS21/topics in emotion analysis/Author_Classification-Team_Lab/data/vocabulary.pickle', 'wb'))

def set_to_vec(pickled_data: set(tuple[tuple[int],str])) -> list[tuple[tuple[int],str]]:
    vec_template = [0 for i in range(len(vocabulary))]
    vectors = list()
    for (document, label) in pickled_data:
        doc_vec = vec_template[:]
        for index in document:
            doc_vec[index] = 1
        vectors.append((doc_vec,label))
    return vectors
