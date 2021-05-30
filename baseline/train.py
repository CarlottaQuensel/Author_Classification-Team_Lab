# -*- coding: utf-8 -*-
from baseline.features import Feature
from baseline import learnFeatures
from baseline.learnFeatures import pointwiseMutualInformation
import numpy as np

'''
after learning x numbers of features, 
initialize x numbers of weights
'''

class MaxEnt():
    '''
    1. Initialize random weights between -2 and +2 (= lambda)
    2. Calculate accuracy of application of these lambdas
    3. Improve accuracy by calculation the derivation and adjusting the lambdas
    4. Return best lambdas for each feature
    '''

    # Initialize a list of random weights
    weights = list()
    features = list()

    def __init__(self, data: list[tuple[tuple[int], str]]=None, class_features: int=None) -> None:
        """Initializing an instance of a Maximum Entropy classifier, if data is already given, 
        then also learning class features

        Args:
            data (list[tuple[tuple[int], str]], optional): Dataset as a list of document vector-label pairs. Defaults to None.
            class_features (int, optional): Number of max learned features per class. Defaults to None.
        """
        if data:
            if class_features:
                self.learnFeatures(data, class_features)
            else:
                self.learnFeatures(data)

    def learnFeatures(self, data: list[tuple[tuple[int], str]], class_features: int=50) -> None:
        """Compute the best features for the classifier based on pointwise mutual information between classes and document features
        in a dataset and save them in a list with a matching list of (untrained) weights. The features are saved as functions for 
        the Max Ent classification and have an associated label and a document property as a number, e.g.
            -> label="Shakespeare" and property=(45, True) (index of "Thou", is in the document).

        Args:
            data (list[tuple[tuple[int], str]]): Dataset consisting of a list of document-label pairs, where the documents are word vectors
            class_features (int): The maximum number of features learned per class, default 50
        """
        # Learning the feature-functions uses PMI and is explained more thoroughly in learnFeatures.py
        self.features = learnFeatures.learnFeatures(data, class_features)

        # Each function has a corresponding weight, so the same number of weights are randomly initialized
        self.weights = [np.random.randint(-2,2) for i in range(len(self.features))]

        # The classifier also has all labels of the training data saved to simplify classification
        self.labels = sorted({feature.label for feature in self.features})

        # After learning the features, information about the number of features and labels is shown
        print(f"The classifier learned {len(self.features)} features for {len(self.labels)} classes.")
    
    def classify(self, document: list[int], in_training: str=False, weights: list[int]=None):
        """The classifier predicts the most probable label from its label set for a document given as a word vector. The Maximum Entropy classifier works
        with a function of a document and label, returning either 1 if if applies or 0 if it doesn't apply to the document-label pair and the function's
        correspinding weight. The probability of a label y given a document is the exponential function of the sum of all weights with applied functions 
        divided (normalized) by the sum of this exponential for all possible labels: p(y|doc) = exp(Σᵢ wᵢ·fᵢ(y,doc)) / Σ(y') exp(Σᵢ wᵢ·fᵢ(y',doc)).
        The classifier computes probabilities for all labels and returns the highest scoring label or, in training, the score itself

        Args:
            document (list[int]): The document to be classified converted into a vector of 0's (word absent) and 1's (word included)
            in_training (str, optional): When computing the derivative in training, the classifier uses the probability itself and not the label. Defaults to False.
            weights (list[int], optional): If testing the accuracy of a specific weight set, custom weights can be used. Defaults to None.

        Returns:
            None: If the custom weights don't match the classifier's number of functions, it cannot compute any probabilities
            str: Outside of the training, the label with the highest probability for the given document is returned
            float: When training, the probability of a specific label is returned for the derivative calculation

        """
        # The default weights to compute the probabilities are the classifier's own weights
        if not weights:
            weights = self.weights
        elif len(weights) != len(self.weights):
            print(f"The classifier needs exactly one weight per function. You used {len(weights)} weights for {len(self.features)} functions.")
            return None

        # When testing the a
        # The numerator of the Max Ent-probability is the exponentioal function of every weight*function with the current label and given document
        numerator = dict()
        for label in self.labels:
            numerator[label] = np.exp(sum([weights[i]*self.features[i].apply(label, document) for i in range(len(self.features))]))
        # As the denominator is the sum of the exponential for all labels, it only depends on the document and is the same for every label
        denominator = sum(numerator.values())
        # The probability of a label then just divides the numerator by the denominator
        p = dict()
        for label in self.labels:
            p[label] = numerator[label] / denominator
        # The classifier either returns the most probable label or in training returns the label's probability
        if in_training:
            return p[in_training]
        return max(sorted(p, reverse=True, key=lambda x: p[x]))


    def accuracy(self, data: list[tuple[tuple[int], str]], weights: list[int]=None) -> float:
        """Compute the basic accuracy of the classifier with a specific weight set as the percentage of correct predictions
        from the overall number of predictions

        Args:
            data (list[tuple[tuple[int], str]]): List of datapoints as pairs of a document vector and a label string
            weights (list[int], optional): Custom weights to compare with the classifier's current weight set. Defaults to None.

        Returns:
            float: The accuracy as the ratio of correct predictions from all predictions
        """
        if not weights:
            weights = self.weights
        gold, predicted = list(), list()
        for doc,label in data:
            gold.append(label)
            predicted.append(self.classify(document=doc, weights=weights))
        tp = 0
        for label_pair in zip(gold,predicted):
            if label_pair[0] == label_pair[1]:
                tp += 1
        return tp/len(gold)


    def train(self, data: list[tuple[tuple[int], str]]):
        ''' 
        Method that trains the model via multivariable linear optimization.
        Since the optimization for the lambda vector needs to happen simultaneous, 
        the iterations stop after it counts 100 (instead of a specific value).

        Args: 
            data (list[tuple[tuple[int], str]], optional): 
            Dataset as a list of document vector-label pairs.
        '''
        
        # compute old accuracy with random weights

        total_iterations = 100 
        n = 1
        old_lambda = self.weights
        new_lambda = list()

        # optimization process
        for n in range(total_iterations):
            # ignore the equation at step 1
            if n != 1:
                old_lambda = new_lambda
                new_lambda = list()
            # iterate over features and call the partial_derivative    
            for i in self.features:
                new_lambda.append(old_lambda[i] - self.partial_derivative(i))

            # check new accuracy

        # calculate the residual to check if the optimization works    
        residual = [x1 - x2 for (x1, x2) in zip(new_lambda, old_lambda)]
        # return weights = new_lambda
  
    def partial_derivative(self, i):
        '''
        Method that computes the derivative of the objective function F
        by substracting the derivative of A from the derivative of B.

        Args:
            index of current lambda for A
            current lambda from the training method for B
            
        Returns:
            derivative of F
        '''

        # calculate first summand 'derivative_A'
        derivative_A = 0                         
        for document in self.documents:
            if self.features[i].apply(self.label, document):
                derivative_A += 1

        # calculate second summmand 'derivative_B'
        derivative_B = 0  

        # iterate through all combinations and check if pair is switched on                     
        for label in self.labels:
            for document in self.documents:
                # probability pλ(y|x)
                # = exp(sum(weights_for_y_given_x)) / exp(sum(weights_for_y'_given_x))
                probability = self.classify(document, in_training=label)
                derivative_B += probability * self.features[i].apply(label, document)

        # calculate derivative of F
        derivative = derivative_A - derivative_B
        return derivative  
