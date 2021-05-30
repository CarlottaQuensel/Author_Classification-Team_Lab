#from train import MaxEnt
from .learnFeatures import learnFeatures
from poetryfoundation_tokenizer import set_to_vec
import pickle
# Set to either Kati's or Carlotta's file path
# path = "/Users/katrin/Desktop/Master/Team Lab/Author_Classification-Team_Lab-1/data/"
path = 'C:/Users/HP Envy/Documents/Uni/Master/SS21/topics in emotion analysis/Author_Classification-Team_Lab/data/'

data = pickle.load(open(f'{path}data.pickle', 'rb'))
vocab = pickle.load(open(f'{path}vocabulary.pickle', 'rb'))
data = set_to_vec(data, vocab)

def train_test_split(data: list[tuple[tuple[int], str]] = data, split: float = 0.8):
    # Training and test data is split in a 80-20 ratio
    train, test = list(), list()
    # Counting poems per author
    authors = dict()
    for doc,label in data:
        try:
            authors[label] += 1
        except KeyError:
            authors[label] = 1
    # For each author save the maximum number of poems in the training set and 
    # the current number of poems in the training set (at initialization 0)
    authors = {author: [int(split*authors[author]), 0] for author in authors}
    # For all datapoints, check if enough poems of this author are in the training data and 
    # decide from that if the datapoint belongs in the training or test data
    for doc, label in data:
        count = authors[label]
        # If there are less poems than the maximum number of this author in the training data,
        if count[0] > count[1]:
            # then put the current datapoint into the training data and update the numbers
            train.append((doc, label))
            authors[label][1] += 1
        # If enough poems are in training, use the datapoint for the test data
        else:
            test.append((doc, label))
    return train, test

learnFeatures(data)
# For later:
'''
classifier = MaxEnt()
train_set, test_set = train_test_split(data)
classifier.learnFeatures(train_set)
classifier.train(train_set)

predicted = list()
gold = [doc[1] for doc in test_set]
for doc in test_set:
    predicted.append(classifier.classify(doc[0]))

eva = Evaluation(gold, predicted)
eva.f_score()
eva.showConfusionMatrix()
'''