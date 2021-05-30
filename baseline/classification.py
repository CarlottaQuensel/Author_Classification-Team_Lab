#from train import MaxEnt
from learnFeatures import learnFeatures
from poetryfoundation_tokenizer import set_to_vec
import pickle
# Set to either Kati's or Carlotta's file path
# path = "/Users/katrin/Desktop/Master/Team Lab/Author_Classification-Team_Lab-1/data/"
path = 'C:/Users/HP Envy/Documents/Uni/Master/SS21/topics in emotion analysis/Author_Classification-Team_Lab/data/'

data = pickle.load(open(f'{path}data.pickle', 'rb'))
vocab = pickle.load(open(f'{path}vocabulary.pickle', 'rb'))
data = set_to_vec(data, vocab)

learnFeatures(data)