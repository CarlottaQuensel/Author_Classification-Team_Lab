# -*- coding: utf-8 -*-
# Author: Carlotta Quensel
from numpy import log
from document import Poem


class Feature():
    """A MaxEnt feature is a binary function of a document-label pair that
    returns 1 if the pair matches the feature's property or 0 otherwise.
    Author: Carlotta Quensel
    """
    label = str()
    property = int()
    form = str()

    def __init__(self, label: str, form: str, doc_property) -> None:
        """Initialization of a maximum entropy feature.

        Args:
            label (str): The author whose weight the function switches on or off
            form (str): Distinguishes the three types of features
            doc_property: Value of three features:
                Token index (int), rhyme scheme (str) and verse range (tuple)

        """
        self.label = label
        self.property = doc_property
        self.form = form
        # Define the right binary function for feature application
        if self.form == "rhyme_scheme":
            # Check if both the given author and poem match the feature's
            # properties with a binary function
            self.switch = lambda x, y: int(
                x == self.label and y.rhyme_scheme == self.property)
        elif self.form == "bow":
            self.switch = lambda x, y: int(
                x == self.label and y.vector[self.property])
        elif self.form == "verse":
            self.switch = lambda x, y: int(
                x == self.label and self.property[0] < y.verse_count <= self.property[1])

    def apply(self, current_label: str, doc: Poem) -> int:
        """Return the binary result of matching the label and document property
        of a maximum entropy feature with the given document and label.

        Args:
            current_label (str):
                The label whose probability the classifier should compute.
            doc (Poem):
                The document instance currently being classified.

        Returns:
            1:  If the feature can be applied to the document and the label
                matches with the probability currently computed.
            0:  If the document doesn't match the property the feature needs
                or another label is currently considered.
        """
        # Call the binary function for the right feature type which converts
        # a boolean into 1 or 0 to switch the feature's weight on or off
        return self.switch(current_label, doc)


def learnFeatures(
    data: list[tuple[Poem, str]], bow_features: int = 30,
    verse_features: bool = True, rhyme_features: int = 5
) -> list[Feature]:
    """Learns the most informative features set of poem-authors pairs.
    The function uses mutual pointwise information to compute the best word and
    rhyme features for each author and the average for the verse number.
    Author: Carlotta Quensel

    Args:
        data (list[tuple[Poem, str]]):
            List of poem-author pairs to learn the features from.
        bow_features (int, optional):
            The number of word features learned per author, defaults to 30.
        verse_features (bool, optional):
            Determines if the verse count is learned or not, defaults to True.
        rhyme_features (int, optional):
            The number of rhyme scheme features learned per author, defaults to 5.

    Returns:
        list[Feature]: Maximum entropy classification features consisting of
            an author label, a document property and the form of the property
            (bag of words, verse number or rhyme scheme)
    """

    features = []
    # Calculate the average verse count per author if wanted by the user
    if verse_features:
        verses = count_verses(data)
        # Convert the average verse ranges into one feature per author
        for bin in verses:
            for author in verses[bin]:
                features.append(
                    Feature(label=author, doc_property=bin, form='verse'))

    # Calculate the pointwise mutual information for token and rhyme features
    bow = bow_pmi(data)
    rhyme = rhyme_pmi(data)
    # Sort PMI scores by relevance to find the most informative features
    for author in bow:
        # Only use the set number of features for words and rhyme scheme
        if bow_features and rhyme_features:
            descending_pmi = sorted([property for property in bow[author]],
                                    reverse=True,
                                    key=lambda x: bow[author][x])[:bow_features]
            descending_rhyme = sorted([property for property in rhyme[author]],
                                      reverse=True,
                                      key=lambda x: rhyme[author][x])[:rhyme_features]
            descending_pmi.extend(descending_rhyme)
        elif rhyme_features:
            descending_pmi = sorted([property for property in rhyme[author]],
                                    reverse=True,
                                    key=lambda x: rhyme[author][x])[:rhyme_features]
        elif bow_features:
            descending_pmi = sorted([property for property in bow[author]],
                                    reverse=True,
                                    key=lambda x: bow[author][x])[:bow_features]
        # Convert the author label and poem properties into Feature objects
        for feature in descending_pmi:
            if type(feature) == str:
                features.append(
                    Feature(label=author, doc_property=feature, form="rhyme_scheme"))
            else:
                features.append(
                    Feature(label=author, doc_property=feature, form="bow"))
    # Return the features sorted by author for a neater look
    # when showing the features
    features.sort(key=lambda x: x.label)
    return features


def bow_pmi(data: list[tuple[Poem, str]]) -> dict[str, dict[int, float]]:
    """Calculate pointwise mutual information between authors and words
    as log( p(author,word) / (p(author)*p(word)) ).
    Author: Carlotta Quensel

    Args:
        data (list[tuple[Poem, str]]):
            List of poem-author pairs, the poem's document vector elements and
            author labels are counted

    Returns:
        dict[str,dict[int,float]]:
            PMI scores sorted first by author, then by word index
    """

    # Corpus and vocabulary size for normalization and easy list apprehension
    doc_number = len(data)
    vocabulary = len(data[0][0].vector)

    # Count the word occurances seperately from authors as the sum of all
    # word vectors (as bag of words uses 0/1, each vector element shows the
    # number of documents containing the word
    c_words = [sum([poem.vector[i] for (poem, author) in data])
               for i in range(vocabulary)]
    # Count words not occuring in documents as the complement given the corpus size
    c_nwords = [doc_number-wordcount for wordcount in c_words]

    # Count the number of authors and author-word combinations in the corpus
    c_authors = {}
    c_authors_words = {}
    for poem, author in data:
        # Handling the first and subsequent occurences of an author
        try:
            c_authors[author] += 1
            # Count the cooccurance of words and authors by summing over the
            # document vectors of the author's poems
            c_authors_words[author] = [c_authors_words[author][i] +
                                       poem.vector[i] for i in range(vocabulary)]
        except KeyError:
            c_authors[author] = 1
            c_authors_words[author] = poem.vector[:]
    # Count authors' occurances without words as the complement of the
    # cooccurance given the number of poems by the author
    # This balances unseen combinations' PMI scores (usually -inf) without
    # the need for normalizing every PMI(x,y) by H(x,y)
    c_authors_nwords = {author: [c_authors[author]-c_authors_words[author][i]
                                 for i in range(vocabulary)] for author in c_authors_words}

    # Normalize the counts with the corpus size to get relative frequencies
    p_words = [wordcount/doc_number for wordcount in c_words]
    p_nwords = [wordcount/doc_number for wordcount in c_nwords]
    p_authors = {author: c_authors[author]/doc_number for author in c_authors}
    p_authors_words = {author: [c_authors_words[author][i] /
                                doc_number for i in range(vocabulary)]
                       for author in c_authors_words}
    p_authors_nwords = {author: [c_authors_nwords[author][i] /
                                 doc_number for i in range(vocabulary)]
                        for author in c_authors_nwords}

    # Calculate the PMI of all author-word combinations
    pmi = dict()
    for author in p_authors:
        pmi[author] = {}
        for word in range(vocabulary):
            # Use the higher score of the cooccurance and the complement as each
            # combination can only form one feature
            score = max(
                log(p_authors_words[author][word] /
                    (p_words[word] * p_authors[author])),
                log(p_authors_nwords[author][word] /
                    (p_nwords[word] * p_authors[author]))
            )
            # Sort the scores first by author label and then by word index
            pmi[author][word] = score

    return pmi


def rhyme_pmi(data: list[tuple[Poem, str]]) -> dict[str, dict[str, float]]:
    """Calculate pointwise mutual information between authors and rhyme schemes
    as log( p(author,rhyme) / (p(author)*p(rhyme)) ).
    Author: Carlotta Quensel

    Args:
        data (list[tuple[Poem, str]]):
            List of poem-author pairs, the Poem.rhyme_scheme property and author
            labels are counted

    Returns:
        dict[dict[str]]: PMI scores sorted first by author, then by rhyme scheme
    """
    # Initialize the counts for separate and joint occurances
    doc_number = len(data)
    c_authors = {}
    c_rhymes = {}
    c_authors_rhymes = {}

    # Count each poem's rhyme scheme and author label separately and as a pair
    for poem, author in data:
        # Address first and subsequent occurance of each feature
        try:
            c_authors[author] += 1
            try:
                c_authors_rhymes[author][poem.rhyme_scheme] += 1
            except KeyError:
                c_authors_rhymes[author][poem.rhyme_scheme] = 1
        except KeyError:
            c_authors[author] = 1
            c_authors_rhymes[author] = {poem.rhyme_scheme: 1}
        try:
            c_rhymes[poem.rhyme_scheme] += 1
        except KeyError:
            c_rhymes[poem.rhyme_scheme] = 1

    # Count inverse of occurances to find very unlikely combinations to
    # balance unseen combinations' PMI scores (usually -inf) without the
    # need for normalizing every PMI(x,y) by H(x,y)
    c_nrhymes = {scheme: doc_number-c_rhymes[scheme] for scheme in c_rhymes}
    c_authors_nrhymes = {}
    # Loop over authors and schemes separately as c_authors_rhymes only contains
    # seen combinations
    for author in c_authors:
        c_authors_nrhymes[author] = {}
        for scheme in c_nrhymes:
            try:
                c_authors_nrhymes[author][scheme] = c_authors[author] - \
                    c_authors_rhymes[author][scheme]
            except KeyError:
                # Count every occurance of an author for unseen combinations
                c_authors_nrhymes[author][scheme] = c_authors[author]
                # Add the unseen combinations as 0 counts (inverse)
                c_authors_rhymes[author][scheme] = 0

    # Normalize the counts with the corpus size to get relative frequencies
    p_authors = {author: c_authors[author]/doc_number for author in c_authors}
    p_rhymes = {scheme: c_rhymes[scheme]/doc_number for scheme in c_rhymes}
    p_nrhymes = {scheme: c_nrhymes[scheme]/doc_number for scheme in c_nrhymes}
    p_authors_rhymes = {author: {
        scheme: c_authors_rhymes[author][scheme]/doc_number for scheme in c_authors_rhymes[author]}
        for author in c_authors_rhymes}
    p_authors_nrhymes = {author: {
        scheme: c_authors_nrhymes[author][scheme]/doc_number for scheme in c_authors_nrhymes[author]}
        for author in c_authors_nrhymes}

    # Calculate the PMI for every possible author-rhyme scheme combination
    pmi = {}
    for author in p_authors:
        pmi[author] = {}
        for scheme in p_rhymes:
            # Use the higher score of the cooccurance and the inverse as each
            # combination can only form one feature
            score = max(log(p_authors_rhymes[author][scheme] / (p_rhymes[scheme] * p_authors[author])),
                        log(p_authors_nrhymes[author][scheme] / (p_nrhymes[scheme] * p_authors[author])))
            # Sort the scores first by author and second by rhyme scheme
            pmi[author][scheme] = score

    return pmi


def count_verses(data: list[tuple[Poem, str]]) -> dict[tuple[int], str]:

    # Save the verse count of each poem sorted by author
    # so that {Author: [verse count of poem1, verse count of poem1, ...], ...}
    count_dictionary = {}
    for poem, Author in data:
        if Author not in count_dictionary:
            count_dictionary[Author] = [poem.verse_count]
        else:
            count_dictionary[Author].append(poem.verse_count)

    # Compute the average verse count of each author
    average_verse_count = {}
    for Author in count_dictionary:
        poems = count_dictionary[Author]
        average_verse_count[Author] = sum(poems) / len(poems)

    # Initiate verse range bins as a dictionary of lower and upper bound tuples
    # for predetermined ranges (see report)
    # Author: Carlotta Quensel during bug fixing,
    #         original code directly below
    bins = {
        (0, 5): [], (5, 10): [], (10, 25): [], (25, 50): [], (50, 75): [], (75, 100): [],
        (100, 150): [], (150, 200): [], (200, 20000): []
    }

    """Original code:
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

    all_lists = [
        list_smaller_5, list_smaller_10, list_smaller_25, list_smaller_50,
        list_smaller_75, list_smaller_100, list_smaller_150, list_smaller_200,
        list_bigger_200
    ]
    all_keys = [
        range(0,5), range(5,10), range(10,25), range(25,50), range(50,75),
        range(75,100), range(100,150), range(150,200), range(200,3500)
    ]
    """

    # Sort each author into the correct range bin according to their average
    for Author in average_verse_count:
        verse_length = average_verse_count[Author]
        if verse_length <= 5:
            bins[(0, 5)].append(Author)
        elif verse_length <= 10 and verse_length > 5:
            bins[(5, 10)].append(Author)
        elif verse_length <= 25 and verse_length > 10:
            bins[(10, 25)].append(Author)
        elif verse_length <= 50 and verse_length > 25:
            bins[(25, 50)].append(Author)
        elif verse_length <= 75 and verse_length > 50:
            bins[(50, 75)].append(Author)
        elif verse_length <= 100 and verse_length > 75:
            bins[(75, 100)].append(Author)
        elif verse_length <= 150 and verse_length > 100:
            bins[(100, 150)].append(Author)
        elif verse_length <= 200 and verse_length > 150:
            bins[(150, 200)].append(Author)
        elif verse_length > 200:
            bins[(200, 20000)].append(Author)

    # Return the authors sorted by the their average verse range
    # Author: Carlotta Quensel during bug fixes,
    #         Orinal code with conversion of lists into dictionary below
    # {(0,5): [author1, author2], (6,10): [author3, author4],...}
    """
    verse_feature_dictionary = {}
    for certain_list in all_lists:
        for key in all_keys:
            verse_feature_dictionary[key] = certain_list
    """

    return bins
