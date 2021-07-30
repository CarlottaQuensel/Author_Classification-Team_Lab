# -*- coding: utf-8 -*-
# Author: Carlotta Quensel
from numpy import log
from document import Poem


class Feature():
    label = str()
    property = int()
    form = str()

    def __init__(self, label: str, form: str, doc_property=None) -> None:
        """Initialization of a maximum entropy feature.
        A MaxEnt feature is a binary function that where a document either has
        a property related to a label or not.

        Args:
            label (str): The author label whose weight the function switches on or off.
            doc_property (optional): Placeholder for detailed features: Token index, 
                rhyme scheme string, poem length and an author only feature (no property needed)

        """
        self.label = label
        self.property = doc_property
        self.form = form
        # Define the right 0/1 function to apply the feature to a poem/author pair
        #if self.form == "apriori":
        #    self.switch = lambda x, y: int(x == self.label)
        if self.form == "rhyme_scheme":
            self.switch = lambda x, y: int(
                x == self.label and y.rhyme_scheme == self.property)
        elif self.form == "bow":
            self.switch = lambda x, y: int(
                x == self.label and y.vector[self.property])
        elif self.form == "verse":
            self.switch = lambda x, y: int(
                x == self.label and self.property[0] < y.verse_count <= self.property[1])

    def apply(self, current_label: str, doc: Poem) -> int:
        """Compares a maximum entropy feature to a current document and label,
        to decide if the MaxEnt classifier will apply the feature and thus
        include a weight for this instance (1) or not (0).

        Args:
            current_label (str): The label whose probability the classifier should compute.
            doc (list[int]): The document instance currently being classified,
            as a bag-of-words feature vector.

        Returns:
            1:  If the feature can be applied to the document and the label
                matches with the probability currently computed.
            0:  If the document doesn't match the property the feature needs
                or another label is currently considered.
        """
        # The feature application calls the correct Poem property by checking which form the feature has
        # The boolean on/off-switch is converted to a number
        # to be multiplied with the weight
        return self.switch(current_label, doc)


def learnFeatures(data: list[tuple[Poem, str]], bow_features: int = 30, verse_features: bool = True, rhyme_features: int = 5) -> list[Feature]:
    """Learns the most informative features for a set of authors from their respective poems. 
    The function uses mutual pointwise information to compute the most relevant word and rhyme 
    features for each author and the average for the verse number.

    Args:
        data (list[tuple[Poem, str]]): List of poems given as a Poem object with poem's author label.
        class_features (int, optional): The number of features learned for each class (author). Defaults to 30.

    Returns:
        list[Feature]: Maximum entropy classification features consisting of an author label, a document property and
            the form of the property (bag of words, verse number or rhyme scheme)
    """

    features = []
    # If the user wants verse count features, calculate the average range of verse count per poem for each
    # author and convert into a feature
    if verse_features:
        verses = count_verses(data)
        for bin in verses:
            for author in verses[bin]:
                features.append(
                    Feature(label=author, doc_property=bin, form='verse'))
    # Calculate the pointwise mutual information for token and verse features
    bow = bow_pmi(data)
    rhyme = rhyme_pmi(data)
    # Sort PMI scores by relevance to find the most informative features
    for author in bow:
        # First sort the word indices and rhyme schemes by the decending value of their PMI score
        if bow_features and rhyme_features:
            descending_pmi = sorted([property for property in bow[author]], reverse=True, key=lambda x: bow[author][x])[:bow_features]
            descending_rhyme = sorted([property for property in rhyme[author]], reverse=True, key=lambda x: rhyme[author][x])[:rhyme_features]
            descending_pmi.extend(descending_rhyme)
        elif rhyme_features:
            descending_pmi = sorted([property for property in rhyme[author]], reverse=True, key=lambda x: rhyme[author][x])[:rhyme_features]
        elif bow_features:
            descending_pmi = sorted([property for property in bow[author]], reverse=True, key=lambda x: bow[author][x])[:bow_features]
        # Instantiate the number of features wanted by the user with the learned word indices and rhyme schemes
        for feature in descending_pmi:
            # Return Max Ent functions as a list of Feature class objects (class description above)
            if type(feature) == str:
                features.append(
                    Feature(label=author, doc_property=feature, form="rhyme_scheme"))
            else:
                features.append(
                    Feature(label=author, doc_property=feature, form="bow"))
    features.sort(key=lambda x: x.label)
    return features


def bow_pmi(data: list[tuple[Poem, str]]) -> dict[dict[int]]:
    doc_number = len(data)
    vocabulary = len(data[0][0].vector)

    c_words = [sum([poem.vector[i] for (poem, author) in data])
               for i in range(vocabulary)]
    c_nwords = [doc_number-wordcount for wordcount in c_words]
    # Counting the frequency of each author and author-word pair
    c_authors = {}
    c_authors_words = {}

    for poem, author in data:
        # Handling the first and subsequent occurences of a label
        try:
            c_authors[author] += 1
            c_authors_words[author] = [c_authors_words[author][i] +
                                       poem.vector[i] for i in range(vocabulary)]
        except KeyError:
            c_authors[author] = 1
            c_authors_words[author] = poem.vector
    c_authors_nwords = {author: [c_authors[author]-c_authors_words[author][i]
                                 for i in range(vocabulary)] for author in c_authors_words}

    p_words = [wordcount/doc_number for wordcount in c_words]
    p_nwords = [wordcount/doc_number for wordcount in c_nwords]
    p_authors = {author: c_authors[author]/doc_number for author in c_authors}
    p_authors_words = {author: [c_authors_words[author][i] /
                                doc_number for i in range(vocabulary)] for author in c_authors_words}
    p_authors_nwords = {author: [c_authors_nwords[author][i] /
                                 doc_number for i in range(vocabulary)] for author in c_authors_nwords}

    pmi = dict()
    for author in p_authors:
        pmi[author] = {}
        for word in range(vocabulary):
            # If the word is more likely not to occur in the author's poems, use the pmi score for word absence
            score = max(log(p_authors_words[author][word] / (p_words[word] * p_authors[author])), log(
                p_authors_nwords[author][word] / (p_nwords[word] * p_authors[author])))
            pmi[author][word] = score

    return pmi


def rhyme_pmi(data: list[tuple[Poem, str]]) -> dict[dict[str]]:
    """Calculate the pointwise mutual information between pairs of author and a poem rhyme scheme
    as log( p(author,rhyme) / (p(author)*p(rhyme)) ) and return the scores sorted by author and rhyme.

    Args:
        data (list[tuple[Poem, str]]): List of Poem and author pairs, where the Poem.rhyme_scheme 
            property is counted with the author label

    Returns:
        dict[dict[str]]: PMI scores sorted first by author, then by rhyme scheme
    """
    doc_number = len(data)

    c_authors = {}
    c_rhymes = {}
    c_authors_rhymes = {}

    for poem, author in data:
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

    c_nrhymes = {scheme: doc_number-c_rhymes[scheme] for scheme in c_rhymes}
    c_authors_nrhymes = {}
    for author in c_authors:
        c_authors_nrhymes[author] = {}
        for scheme in c_nrhymes:
            try:
                c_authors_nrhymes[author][scheme] = c_authors[author] - \
                    c_authors_rhymes[author][scheme]
            except KeyError:
                c_authors_nrhymes[author][scheme] = c_authors[author]
                c_authors_rhymes[author][scheme] = 0

    p_authors = {author: c_authors[author]/doc_number for author in c_authors}
    p_rhymes = {scheme: c_rhymes[scheme]/doc_number for scheme in c_rhymes}
    p_nrhymes = {scheme: c_nrhymes[scheme]/doc_number for scheme in c_nrhymes}
    p_authors_rhymes = {author: {
        scheme: c_authors_rhymes[author][scheme]/doc_number for scheme in c_authors_rhymes[author]} for author in c_authors_rhymes}
    p_authors_nrhymes = {author: {
        scheme: c_authors_nrhymes[author][scheme]/doc_number for scheme in c_authors_nrhymes[author]} for author in c_authors_nrhymes}

    pmi = {}
    for author in p_authors:
        pmi[author] = {}
        for scheme in p_rhymes:
            score = max(log(p_authors_rhymes[author][scheme] / (p_rhymes[scheme] * p_authors[author])), log(
                p_authors_nrhymes[author][scheme] / (p_nrhymes[scheme] * p_authors[author])))
            pmi[author][scheme] = score

    return pmi


def count_verses(data: list[tuple[Poem, str]]):

    # iterate over authors in dictionary
    # and count verses,  create new dictionary
    # so that {Author: [count of verse1, count of verse2, ...], ...}
    count_dictionary = {}
    for poem, Author in data:
        if Author not in count_dictionary:
            count_dictionary[Author] = [poem.verse_count]
        else:
            count_dictionary[Author].append(poem.verse_count)

    # compute average count of verses per author
    # so that {Author: [average count of verses], ...}
    average_verse_count = {}
    for Author in count_dictionary:
        poems = count_dictionary[Author]
        average_verse_count[Author] = sum(poems) / len(poems)

    # Bins are initiated as a dictionary of the pairs of lower and upper bound of verse count
    # and the lists of all authors fitting into this bin
    # (Author: Carlotta Quensel during bug fixing, originally individual lists, saved in global list)
    bins = {
        (0, 5): [], (5, 10): [], (10, 25): [], (25, 50): [], (50, 75): [], (75, 100): [],
        (100, 150): [], (150, 200): [], (200, 20000): []
    }

    """# Set bins
        list_smaller_5 = []
        list_smaller_10 = []
        list_smaller_25 = []
        list_smaller_50 = []
        list_smaller_75 = []
        list_smaller_100 = []
        list_smaller_150 = []
        list_smaller_200 = []
        list_bigger_200 = []

        all_lists = [list_smaller_5, list_smaller_10, list_smaller_25, list_smaller_50, list_smaller_75, list_smaller_100, list_smaller_150, list_smaller_200, list_bigger_200]
        all_keys = [range(0,5), range(5,10), range(10,25), range(25,50), range(50,75), range(75,100), range(100,150), range(150,200), range(200,3500)]"""

    # Iterate over authors in the dictionary
    # and their average verse length, in order
    # to assign them to the bins.
    # so that list_smaller_X = [Author1, Author2, ...]
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

    # Return all authors sorted into the right bins
    # (Author: Carlotta Quensel during bug fixes, originally with additional conversion of lists into dictionary)
    # {(0,5): [author1, author2], (6,10): [author3, author4],...}
    """verse_feature_dictionary = {}
        for certain_list in all_lists:
            for key in all_keys:
                verse_feature_dictionary[key] = certain_list"""

    return bins
