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
        # The function calls for the label to match the function and
        # for the document to include a word or property.
        if self.form == "rhyme_scheme":
            switch = (self.label == current_label and self.property ==
                      doc.rhyme_scheme)
        elif self.form == "bow":
            switch = (self.label ==
                      current_label and doc.vector[self.property])
        elif self.form == "verse":
            switch = (self.label == current_label and self.property[0] <
                      doc.verse_count <= self.property[1])
        elif self.form == "apriori":
            switch = (self.label == current_label)
        # The boolean on/off-switch is converted to a number
        # to be multiplied with the weight
        return int(switch)


def learnFeatures(data: list[tuple[Poem, str]], class_features: int = 50) -> list[Feature]:
    """Learns the most informative features for a set of authors from their respective poems. 
    The function uses mutual pointwise information to compute the most relevant word and rhyme 
    features for each author and the average for the verse number.

    Args:
        data (list[tuple[Poem, str]]): List of poems given as a Poem object with poem's author label.
        class_features (int, optional): The number of features learned for each class (author). Defaults to 50.

    Returns:
        list[Feature]: Maximum entropy classification features consisting of an author label, a document property and
            the form of the property (bag of words, verse number or rhyme scheme)
    """

    #verses = count_verses(data)
    features = list()
    #for bin in verses:
    #        for author in verses[bin]:
    #            features.append(
    #                Feature(label=author, doc_property=bin, form='verse'))
    # Calculate the pointwise mutual information for token and verse features
    bow, rhyme = pmi(data)
    # For x learned class features, one is used for the a priory weight of an author, one is the author's
    # verse average, leaving x-2 token/rhyme features from which the highest PMI scores are used separately
    # to balance between the many token counts and sparse rhyme scheme feature (one per poem)
    class_features = class_features-1
    rhyme_features = int((class_features)*0.3)
    bow_features = class_features-rhyme_features
    # Sort PMI scores by relevance to find the most informative features
    for author in bow:
        features.append(Feature(label=author, form="apriori"))
        # First sort the word indices and rhyme schemes by the decending value of their PMI score
        # The absolute is used for sorting to allow for features with negative weights (unlikely author-word combinations)
        decending_bow = sorted([property for property in bow[author]], reverse=True, key=lambda x: abs(
            bow[author][x]))[:bow_features]
        decending_rhyme = sorted([property for property in rhyme[author]], reverse=True, key=lambda x: abs(
            rhyme[author][x]))[:rhyme_features]
        decending_pmi = decending_bow[:]
        decending_pmi.extend(decending_rhyme)
        # Instantiate one less feature than wanted by the author with the learned word indices and rhyme schemes
        # (as there are only 24 possible rhyme schemes, but as many words as the size of the vocabulary,
        # the split might not be even)
        # (the last feature for each author is the average verse number already calculated)
        for feature in decending_pmi:
            # Return Max Ent functions as a list of Feature class objects (class description above)
            if type(feature) == str:
                features.append(
                    Feature(label=author, doc_property=feature, form="rhyme_scheme"))
            else:
                features.append(
                    Feature(label=author, doc_property=feature, form="bow"))
    features.sort(key=lambda x: x.label)
    return features


def pmi(data: list[tuple[Poem, str]]) -> tuple[dict[dict[int]], dict[dict[str]]]:
    """Calculate the pointwise mutual information of a label y and text property x as
    log( p(x,y) / (p(x)*p(y)) ) between author labels and the bag of words and rhyme scheme
    property of the given poem list.

    Args:
        data (list[tuple[Poem, str]]): The data to calculate the PMI from as pairs of a poem and
        author. The poems' word vector and rhyme scheme class feature (see document.py) are used

    Returns:
        tuple(dict[dict[int]], dict[dict[str]]): Return two sets of PMI scores for word (word vector index)
        and rhyme (scheme string) features, sorted first by author, then by feature.
    """
    doc_number = len(data)
    vocabulary = len(data[0][0].vector)

    # Initializing the counts for the different features (bag of words/verse number/rhyme scheme) and authors
    # and the feature-label combination counts
    c_authors = dict()
    c_rhymes = dict()
    c_words = [0 for i in range(vocabulary)]
    c_author_words = dict()
    c_author_rhymes = dict()

    # Iteratively counting each poem's features and author
    for (poem, author) in data:
        # The word counts for the whole poetry collection are added up
        c_words = [c_words[i] + poem.vector[i] for i in range(vocabulary)]
        # The author counts and all combined counts for author and property are added up
        # Unseen authors are addressed by instantiating their counts instead of incrementing them
        try:
            c_authors[author] += 1
            c_author_words[author] = [c_author_words[author]
                                      [i] + poem.vector[i] for i in range(vocabulary)]
            # If the author was already seen before, the current rhyme scheme might still be new
            try:
                c_author_rhymes[author][poem.rhyme_scheme] += 1
            except KeyError:
                c_author_rhymes[author][poem.rhyme_scheme] = 1
        except KeyError:
            c_authors[author] = 1
            c_author_rhymes[author] = {poem.rhyme_scheme: 1}
            c_author_words[author] = poem.vector
        # The counts for the rhyme schemes are added up separately.
        try:
            c_rhymes[poem.rhyme_scheme] += 1
        except KeyError:
            c_rhymes[poem.rhyme_scheme] = 1

    # The counts are converted into probabilities by dividing them by the total number
    # of poems in the collection:
    # -> Individual probabilities for the denominator of PMI
    p_words = [c_words[i]/doc_number for i in range(vocabulary)]
    p_rhymes = {rhyme: c_rhymes[rhyme]/doc_number for rhyme in c_rhymes}
    p_authors = {author: c_authors[author]/doc_number for author in c_authors}
    # -> Combined probabilities for the numerator of PMI (smothed for the token features to avoid
    #    -infinity pmi values for unseen combinations)
    p_author_words = {author: [(c_author_words[author][i]+1) /
                               (doc_number+1) for i in range(vocabulary)] for author in c_author_words}
    p_author_rhymes = {author: {
        scheme: (c_author_rhymes[author][scheme]+1)/(doc_number+1) for scheme in c_author_rhymes[author]} for author in c_author_rhymes}

    # Calculating PMI for all word/author combinations as log( P(word,author) / P(word)*P(author) )
    for index in range(vocabulary):
        if p_words[index] == 0:
            print(index, "count:", c_words[index])
    bow_pmi = {author: {i: log(p_author_words[author][i] / (p_authors[author]*p_words[i]))
                        for i in range(vocabulary)} for author in p_author_words}
    # Calculating the PMI for rhyme scheme and author combinations
    rhyme_pmi = {}
    for author in p_author_rhymes:
        rhyme_pmi[author] = {}
        for scheme in p_rhymes:
            try:
                rhyme_pmi[author][scheme] = log(
                    p_author_rhymes[author][scheme] / (p_authors[author]*p_rhymes[scheme]))
            except KeyError:
                # Add-one smoothing for unseen author-scheme combinations
                rhyme_pmi[author][scheme] = log((1/(doc_number+1)) / (p_authors[author]*p_rhymes[scheme]))
    # The two property's PMI scores are given to the feature learning function
    return bow_pmi, rhyme_pmi


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
            (0,5): [], (5,10): [], (10, 25): [], (25, 50): [], (50, 75): [], (75, 100): [],
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
                bins[(10,25)].append(Author)
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
