# -*- coding: utf-8 -*-
# Author: Carlotta Quensel
from nltk import WordPunctTokenizer
import pronouncing
import re


class Poem():
    """Given a poem as one string, Poem extracts and saves the tokens as a
    word vector, the verses and verse count and the rhyme scheme.
    Author: Carlotta Quensel
    """
    verses = list()
    verse_count = int()
    rhyme_scheme = str()
    vector = list()

    def __init__(self, poem: str, vocab: dict[str, int]):
        """Extract a poem's features and save them as class properties.
        Author: Carlotta Quensel

        Args:
            poem (str): A poem given as one string
            vocab (dict[str, int]): The vocabulary against which the 
                poem's tokens are checked to form the word vector
        """
        # Extract the poem's features with the corresponding method and save them
        # as properties of the new object
        self.raw = poem
        self.getVerses()
        self.getRhymes()
        self.getVector(vocab)

    def getVerses(self):
        """Split the unaltered string of the poem into lines and those into
        tokens to get the verses and verse count.
        Author: Katrin Schmidt
        """
        lines = self.raw.splitlines()

        # Count number of verses
        self.verse_count = len(lines)

        # Tokenize the poem verse by verse to allow getRhymes access to
        # each verse's last word
        tokenizer = WordPunctTokenizer()
        self.verses = [[word.lower() for word in tokenizer.tokenize(
            lines[i].strip())] for i in range(len(lines))]
        self.tokens = self.verses[0][:]
        for i in range(1, len(self.verses)):
            self.tokens.extend(self.verses[i])

    def getRhymes(self):
        """Build the poem's rhyme scheme from the last words of the first four verses.
        Author: Carlotta Quensel
        """
        # Before checking for rhymes, remove sentence final punctuation
        # and blank lines to ensure correct rhyme detection
        punctuation = "[^0-9A-Za-z]"
        i = 0
        self.stanzas = 1
        # Use while to iterate over the verses since the deletion of lines
        # changes the length of the list
        while i < len(self.verses):
            if not len(self.verses[i]):
                # Delete blank lines/stanza delimiters
                self.verses.pop(i)
                self.stanzas += 1
            else:
                # Delete verse final punctuation so it won't mess with the rhyme scheme
                self.verses[i][-1] = re.sub(punctuation,
                                            "", self.verses[i][-1])
                if not len(self.verses[i][-1]):
                    self.verses[i].pop()
                    # Lines only consisting of punctuation are deleted for the same reason
                    if not len(self.verses[i]):
                        self.verses.pop(i)
                        continue
                i += 1

        # The rhyme scheme is constructed with the first four sentences of the poem
        rhymes = ["a", "b", "c", "d"]
        # Initialize unmatched verses and rhyme scheme bins
        verses = {i for i in range(len(self.verses))}
        bins = {}
        # Look at the first four verses
        for i in range(min(len(self.verses), 4)):
            # Each verse that is not yet matched to a rhyme scheme is
            # matched to the next rhyme placeholder ("a", "b", ...) and deleted
            # from the unmatched verses
            if i in verses:
                verses.remove(i)
                schema = rhymes.pop(0)
                bins[i] = schema
                # The subsequent verses are checked against the current verse
                for j in range(i+1, min(len(self.verses), 4)):
                    # For the last word of the two current verses check if
                    # they are identical or if either one is in the rhyme set
                    # of the other (this is necessary as pronouncing uses a
                    # dictionary to check rhymes, whose entries are not always
                    # identical sets)
                    if self.verses[j][-1] in pronouncing.rhymes(self.verses[i][-1]) \
                            or self.verses[i][-1] in pronouncing.rhymes(self.verses[j][-1]) \
                            or self.verses[i][-1] == self.verses[j][-1]:
                        # Assign an unmatched subsequent verse to the current rhyme scheme
                        # and delete it from the unmatched set
                        if j in verses:
                            bins[j] = schema
                            verses.remove(j)
                        # If the subsequent verse is already matched to a rhyme
                        # scheme, this means that the pronouncing rhyme sets of
                        # a previous verse included it but not the current verse
                        else:
                            # Use the older rhyme scheme instead of the current one
                            schema_j = bins[j]
                            bins[i] = schema_j
                            # Reassign all verses with the current rhyme scheme
                            # to the older one (including the current verse)
                            for k in bins:
                                if bins[k] == schema:
                                    bins[k] = schema_j
                            # Rejoin the current rhyme scheme to the rhyme placeholders
                            schema_j = [schema_j]
                            schema_j.extend(rhymes)
                            rhymes = schema_j
        # Extend poems under four lines with dummy placeholders
        if len(self.verses) < 4:
            for i in range(len(self.verses), 4):
                bins[i] = "x"
        # Construct the rhyme scheme by concatenating the rhyme placeholders
        # of the first four lines
        self.rhyme_scheme = bins[0]+bins[1]+bins[2]+bins[3]

    def getVector(self, vocab: dict[str, int]):
        """Build a bag-of-words vector from the tokens in the poem 
        by checking it against the given vocabulary

        Args:
            vocab (dict[str, int]): 
                Vocabulary assigning tokens to indices in
                a bag-of-words vector
        """
        # Initialize the poems vector with no words
        vec = [0 for i in range(len(vocab))]
        for word in self.tokens:
            try:
                # All tokens of the poem that are part of the vocabulary
                # are marked as included by setting the respective vector
                # element to 1F
                vec[vocab[word]] = 1
            except KeyError:
                pass
        self.vector = vec
