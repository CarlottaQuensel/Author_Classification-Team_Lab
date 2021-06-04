# -*- coding: utf-8 -*-
from features import Feature
from learnFeatures import *
#from baseline.learnFeatures import pointwiseMutualInformation
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
        """
        Author: Katrin Schmidt
        Initializing an instance of a Maximum Entropy classifier, if data is already given, 
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

    def learnFeatures(self, data: list[tuple[tuple[int], str]], class_features: int=50, vocabulary: dict[str] = None) -> None:
        """
        Author: Carlotta Quensel
        Compute the best features for the classifier based on pointwise mutual information between classes and document features
        in a dataset and save them in a list with a matching list of (untrained) weights. The features are saved as functions for 
        the Max Ent classification and have an associated label and a document property as a number, e.g.
            -> label="Shakespeare" and property=45 (index of "Thou", is in the document).

        Args:
            data (list[tuple[tuple[int], str]]): Dataset consisting of a list of document-label pairs, where the documents are word vectors
            class_features (int): The maximum number of features learned per class, default 50
            vocabulary (dict[str]): The assignment of words to word vector indices used in the given data set
        """
        if vocabulary:
            self.vocabulary = vocabulary

        # Learning the feature-functions uses PMI and is explained more thoroughly in learnFeatures.py
        self.features = learnFeatures(data, class_features)

        # Each function has a corresponding weight, so the same number of weights are randomly initialized
        self.weights = [np.random.randint(-2,2) for i in range(len(self.features))]

        # The classifier also has all labels of the training data saved to simplify classification
        self.labels = sorted({feature.label for feature in self.features})

        # After learning the features, information about the number of features and labels is shown
        print(f"The classifier learned {len(self.features)} features for {len(self.labels)} classes.")
    
    def classify(self, document: list[int], in_training: str=False, weights: list[int]=None):
        """
        Author: Carlotta Quensel
        The classifier predicts the most probable label from its label set for a document given as a word vector. The Maximum Entropy classifier works
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
        """
        Author: Carlotta Quensel
        Compute the basic accuracy of the classifier with a specific weight set as the percentage of correct predictions
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


    def train(self, data: list[tuple[tuple[int], str]], min_improvement: float = 0.001, trace: bool = False):
        ''' 
        Author: Katrin Schmidt, trace and accuracy Carlotta Quensel
        Method that trains the model via multivariable linear optimization.
        Since the optimization for the lambda vector needs to happen simultaneous, 
        the iterations stop after it counts 100 (instead of a specific value).

        Args: 
            data (list[tuple[tuple[int], str]]): 
                Dataset as a list of document vector-label pairs.
            min_improvement (float, optional): The minimum accuracy improvement needed for a new iteration of optimization. Defaults to 0.0001
            trace (boolean, optional): Switch on to track the development of the accuracy during the iterative process.
        '''
        # Training as an optimization process for the weights
        # Either with fixed number of iterations or until the improvement drops below a threshold (currently at least 0.1% improvement)
        
        # compute old accuracy with random weights
        old_accuracy = 0
        new_accuracy = self.accuracy(data)
        if trace:
            print(f"With randomized weights for the features, the classifier's accuracy is {new_accuracy}.")
            acc = [new_accuracy]
        new_lambda = list()

        if trace:
            i = 0
        while new_accuracy - old_accuracy >= min_improvement:
        #total_iterations = 100
        #for n in range(total_iterations):
            # Don't reassign the new weights in the first step (no new weights yet)
            if len(new_lambda):
                self.weights = new_lambda
                if trace:
                    print(f"In the {i}. iteration, the classifier's accuracy improved by {new_accuracy - old_accuracy} and is now at {new_accuracy}.")
            # Perform stochastic gradient descent by iterating over features and computing the partial_derivative
            # and save the updated weights temporarily to first make sure that they improve the classifier
            new_lambda = [self.weights[i] - self.partial_derivative(data, i) for i in range(len(self.features))]

            # Calculate the accuracy with the new weights to check its improvement
            new_accuracy = self.accuracy(data, new_lambda)
            if trace:
                acc.append(new_accuracy)

        if trace:
            print(f"The training consisted of {i} optimization steps in which the accuracy improved from {acc[0]} to {acc[-1]}.")
            # If the user wants to track the training process, the accuracy scores are returned to potentially plot the improvement
            return acc

  
    def partial_derivative(self, data: list[tuple[tuple[int],str]], i: int) -> float:
        '''
        Author: Katrin Schmidt
        Method that computes the derivative of the objective function F
        by substracting the derivative of A from the derivative of B.

        Args:
            data (list[tuple[tuple[int],str]]): Dataset as a list of document vector-label pairs.
            i (int): Index of current lambda for the partial derivative ∂A/∂λi - ∂B/∂λi w.r.t. λi
            
        Returns:
            float: Partial derivative of the maximum entropy function F w.r.t. λi
        '''

        # calculate first summand 'derivative_A' as ∂A/∂λi = Σ(y,x) fλ(y|x) or the number of correctly recognized document-label pairs
        derivative_A = sum([self.features[i].apply(label, document) for document, label in data])

        # calculate second summmand 'derivative_B'
        derivative_B = 0  

        # Iterate through all label-document combinations and add up the probabilities for the switched on pairs
        for label in self.labels:
            for document, gold_label in data:
                # probability pλ(y|x) = exp(sum(weights_for_y_given_x)) / exp(sum(weights_for_y'_given_x))
                # ∂B/∂λi = Σ(y,x)Σy'        pλ(y'|x)                       *         fλ(y'|x)
                derivative_B += self.classify(document, in_training=label) * self.features[i].apply(label, document)

        # calculate derivative of F
        derivative = derivative_A - derivative_B
        return derivative  
