from nltk import WordPunctTokenizer
import pronouncing
import re

class Poem():
    verses = list()
    rhyme_scheme = str()
    vector = list()

    def __init__(self, poem: str, vocab: dict[str]):
        self.raw = poem
        self.getVerses()
        self.getRhymes()
        self.getVector(vocab)
    

    def getVerses(self):
        lines = self.raw.splitlines()

        # Count number of verses
        self.verse_count = len(lines)

        tokenizer = WordPunctTokenizer()
        self.verses = [[word.lower() for word in tokenizer.tokenize(lines[i].strip())] for i in range(len(lines))]
        self.tokens = self.verses[0][:]
        for i in range(1,len(self.verses)):
            self.tokens.extend(self.verses[i])
    
    def getRhymes(self):
        punctuation = "[^0-9A-Za-z]"
        i = 0
        self.stanzas = 1
        while i < len(self.verses):
            if not len(self.verses[i]):
                # Delete blank lines/stanza delimiters
                self.verses.pop(i)
                self.stanzas += 1
            else:
                # Delete verse final punctuation so it won't mess with the rhyme scheme
                self.verses[i][-1] = re.sub(punctuation, "", self.verses[i][-1])
                if not len(self.verses[i][-1]):
                    self.verses[i].pop()
                    # Lines only consisting of punctuation are deleted for the same reason
                    if not len(self.verses[i]):
                        self.verses.pop(i)
                        continue
                i += 1
        # The rhyme scheme is constructed with the first four sentences of the poem
        rhymes = ["a", "b", "c", "d"]
        verses = {i for i in range(len(self.verses))}
        bins = {}
        for i in range(min(len(self.verses), 4)):
            if i in verses:
                verses.remove(i)
                schema = rhymes.pop(0)
                bins[i] = schema
                for j in range(i+1,len(self.verses)):
                    if self.verses[j][-1] in pronouncing.rhymes(self.verses[i][-1]) or self.verses[i][-1] in pronouncing.rhymes(self.verses[j][-1]) or self.verses[i][-1] == self.verses[j][-1]:
                        if j in verses:
                            bins[j] = schema
                            verses.remove(j)
                        else:
                            schema_j = bins[j]
                            bins[i] = schema_j
                            for k in bins:
                                if bins[k] == schema:
                                    bins[k] = schema_j
                            schema_j = [schema_j]
                            schema_j.extend(rhymes)
                            rhymes = schema_j
        # Poems under four lines are extended with dummy schemes
        if len(self.verses) < 4:
            for i in range(len(self.verses), 4):
                bins[i] = "x"
        self.rhyme_scheme = bins[0]+bins[1]+bins[2]+bins[3]

    def getVector(self, vocab: dict[str]):
        vec = [0 for i in range(len(vocab))]
        for word in self.tokens:
            try:
                vec[vocab[re.sub("[^0-9A-Za-z]","", word)]] = 1
            except KeyError:
                pass
        self.vector = vec
            