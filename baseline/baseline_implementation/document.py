from nltk import WordPunctTokenizer

class Document():

    def __init__(self, poem: str, vocab: dict[str]):
        self.raw = poem
        tokenizer = WordPunctTokenizer()
        lines = poem.splitlines()
        