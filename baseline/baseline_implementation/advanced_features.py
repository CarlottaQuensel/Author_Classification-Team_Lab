import re
import pronouncing
from nltk import WordPunctTokenizer


def getRhymes(poem):
	tok = WordPunctTokenizer()
	stanzas = poem.splitlines()
	punctuation = "[^0-9A-Za-z]"
	i = 0
	while i < len(stanzas):
		stanzas[i] = stanzas[i].strip()
		if not len(stanzas[i]):
			stanzas.pop(i)
		else:
			stanzas[i] = tok.tokenize(stanzas[i])
			stanzas[i][-1] = re.sub(punctuation, "", stanzas[i][-1])
			if not len(stanzas[i][-1]):
				   stanzas[i].pop()
			i += 1
	rhymes = ["a", "b", "c", "d"]
	verses = {i for i in range(len(stanzas))}
	bins = []
	for i in range(max(len(stanzas), 4)):
		if i in verses:
			verses.remove(i)
			schema = rhymes.pop(0)
			bins.append((i, schema))
			for j in range(i+1,len(stanzas)):
				if stanzas[j][-1] in pronouncing.rhymes(stanzas[i][-1]) or stanzas[i][-1] == stanzas[j][-1]:
					bins.append((j,schema))
					verses.remove(j)
	bins.sort()
	rhyme_scheme = bins[0][1]+bins[1][1]+bins[2][1]+bins[3][1]
	return rhyme_scheme

def getBagOfWords(poem):
	tok = WordPunctTokenizer()
	tokens = tok.tokenize(poem)
	punctuation = "[^0-9A-Za-z]"
	i = 0
	while i < len(tokens):
		tokens[i] = re.sub(punctuation, "", tokens[i]).lower()
		if not len(tokens[i]):
			tokens.pop(i)
		else:
			i += 1