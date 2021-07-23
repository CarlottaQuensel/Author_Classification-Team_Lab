from nltk import WordPunctTokenizer
import pronouncing
import re

class Poem():
    verses = list()
    rhyme_scheme = str()
    vector = list()

    def __init__(self, poem: str, vocab: dict[str]):
        self.raw = poem
        tokenizer = WordPunctTokenizer()
        lines = poem.splitlines()
        self.verses = [[word.lower() for word in tokenizer.tokenize(lines[i].strip())] for i in range(len(lines))]
        self.tokens = self.verses[0]
        for i in range(1,len(self.verses)):
            self.tokens.extend(self.verses[i])
        self.getRhymes()
        self.getVector(vocab)
    
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
                i += 1
        # The rhyme scheme is constructed with the first four sentences of the poem
        rhymes = ["a", "b", "c", "d"]
        verses = {1,2,3,4}
        bins = {}
        for i in range(min(len(self.verses), 4)):
            if i in verses:
                verses.remove(i)
                schema = rhymes.pop(0)
                bins[i] = schema
                for j in range(i+1,len(self.verses)):
                    if self.verses[j][-1] in pronouncing.rhymes(self.verses[i][-1]) or self.verses[i][-1] == self.verses[j][-1]:
                        bins[j] = schema
                        verses.remove(j)
        self.rhyme_scheme = bins[0]+bins[1]+bins[2]+bins[3]

    def getVector(self, vocab: dict[str]):
        vec = [0 for i in range(len(vocab))]
        for word in self.tokens:
            try:
                vec[vocab[word]] = 1
            except KeyError:
                pass
        self.vector = vec
            