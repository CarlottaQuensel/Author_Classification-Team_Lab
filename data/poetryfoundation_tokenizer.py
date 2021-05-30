from nltk.tokenize import WordPunctTokenizer
import pandas
import pickle
# TODO: think about replacing '\n' by ' .' --> if ' .' is followed by '.'

# path = "/Users/katrin/Desktop/Master/Team Lab/Author_Classification-Team_Lab-1/data/"
path = "C:/Users/HP Envy/Documents/Uni/Master/SS21/topics in emotion analysis/Author_Classification-Team_Lab/data/"


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
            dictionary_of_poems[row[1]] = [str(row[4]).replace('\n', ' ')] #[str([row[4]]).strip('\n')]
        else:
            dictionary_of_poems[row[1]].append(str(row[4]).replace('\n', ' '))

    # create a smaller data_frame for testing
    # data_frame_head = {k: dictionary_of_poems[k] for k in list(dictionary_of_poems)[2:6]}


# ----------------------------------------------------
# --------------------TOKENIZATION--------------------
# ----------------------------------------------------
# tokenizing the data with nltk
tokenize_punctuation = WordPunctTokenizer()
tokenized_dictionary = {}

# iterate over Authors in dictionary
for author in dictionary_of_poems:

    # iterate over each poem per Author
    for poem in dictionary_of_poems[author]:
        tokenized_poem = tokenize_punctuation.tokenize(poem)
        
        # create a new dictionary with tokenized poems
        if author not in tokenized_dictionary:
            tokenized_dictionary[author] = [tokenized_poem]
        else:
            tokenized_dictionary[author].append(tokenized_poem)
pickle.dump(tokenized_dictionary, open('tokenized_dictionary.pickle', 'wb'))
#print(tokenized_dictionary)


# -----------------------------------------------------------
# --------------------BUILDING VOCABULARY--------------------
# -----------------------------------------------------------
def build_vocab(data: dict[str] = tokenized_dictionary, min_poems: int = 30, max_authors: int = 0) -> dict[str]:
    """Take labelled documents and build a vocabulary of the included word types.

    Args:
        data (dict[str]): 
            Documents sorted into author categories as lists of tokenized words
            {'Author': [[poem1], [poem2],...], 'Author2': [...]} with [poem] = ['word1', 'word2', ...]
        min_poems (int, optional): 
            Build vocabulary only from poems of authors who wrote at least this many poems. Limits the number 
            of classes for a classifier to learn by excluding classes with few datapoints. Defaults to 30.
        max_authors (int, optional): 
            Build vocabulary from poems of only this many authors. 
            Sets the number of classes for a classifier to learn. Defaults to 0.

    Returns:
        dict[str]: Vocabulary as a dictionary with words pointing to their index in the document vectors
    """
    authors = sorted(data.keys(), reverse=True, key=lambda x: len(data[x]))
    # Set maximum amount of author labels if given
    if max_authors:
        authors = authors[:max_authors]
    vocabulary = set()
    author_list = list()
    # Iterate over the tokenized poems by author and include the tokens of the most prolific authors into the vocabulary
    for author in authors:
        # Exclude the label 'anonymous' from the most prolific authors as it is not a name for an individual author
        if len(data[author]) >= min_poems and author != 'Anonymous':
            author_list.append(author)
            for poem in data[author]:
                # Save the tokens as a set first to convert into types (i.e. delete duplicates)
                types = set(poem)
                vocabulary = vocabulary.union(types)

    # Convert the unordered set of types into a table of all types and their index in an alphabetical list
    vocabulary = {word: i for i, word in enumerate(sorted(vocabulary))}
    # Use this list of words as a indexing guide for the word vectors
    author_list = sorted(author_list)
    pickle.dump(vocabulary, open(f'{path}vocabulary.pickle', 'wb'))
    return vocabulary, author_list

# 39 named authors with at least 30 poems (-> 1576 poems):
#       'Alfred, Lord Tennyson', 'Algernon Charles Swinburne', 'Alice Notley', 'Ben Jonson', 'Billy Collins', 'Carl Sandburg', 
#      'Christina Rossetti', 'David Ferry', 'Dean Young', 'Edgar Lee Masters', 'Edmund Spenser', 'Emily Dickinson', 
#       'Frank Stanford', 'George Herbert', 'Gwendolyn Brooks', 'Henry Wadsworth Longfellow', 'Jane Hirshfield', 'John Ashbery', 
#      'John Donne', 'John Keats', 'John Milton', 'Kahlil Gibran', 'Kay Ryan', 'Percy sshe Shelley', 'Rae Armantrout', 
#       'Robert Browning', 'Robert Herrick', 'Samuel Menashe', 'Samuel Taylor Coleridge', 'Sir Philip Sidney', 'Thomas Hardy', 
#      'W. S. Di Piero', 'W. S. Merwin', 'Walt Whitman', 'William Blake', 'William Butler Yeats', 'William Shakespeare', 
#       'William Wordsworth', 'Yusef Komunyakaa'
# Vocabulary = 38814 word types
vocabulary, author_list = build_vocab()


# -----------------------------------------------
# --------------------VECTORS--------------------
# -----------------------------------------------

# word vectors (lists of 0 and 1 or set of indexes with 1's), e.g.:
#       words -> {I=1, think=2, but=3, am=4, a=5, cat=6, can=7, speak=8}
#       doc1 = ["I", "think"] -> used as: [1,1,0,0,0,0,0,0] / saved as: {1,2}
#       doc2 = ["I", "am", "a", "cat"] -> used as: [1,0,0,1,1,1,0,0] / saved as: {1,4,5,6}

data = set()
for author in author_list:
    for poem in tokenized_dictionary[author]:
        vec = set()
        for token in poem:
            vec.add(vocabulary[token])
        data.add((tuple(vec), author))

pickle.dump(data, open(f'{path}data.pickle', 'wb'))
pickle.dump(vocabulary, open(f'{path}vocabulary.pickle', 'wb'))



# --------------------------------------------------------
# --------------------UNPICKLE TO VECS--------------------
# --------------------------------------------------------
def set_to_vec(pickled_data: set[tuple[tuple[int],str]], vocab: dict[str]=vocabulary) -> list[tuple[tuple[int],str]]:
    # Initialize all document vectors as not including any words from the vocabulary
    vec_template = [0 for i in range(len(vocab))]
    vectors = list()
    for (document, label) in pickled_data:
        doc_vec = vec_template[:]
        # The entry of every vocabulary word that is in the document is set to 1
        for index in document:
            doc_vec[index] = 1
        # The data for further use is structured as a list of pairs from each document's word vector and label
        vectors.append((doc_vec,label))
    return vectors
