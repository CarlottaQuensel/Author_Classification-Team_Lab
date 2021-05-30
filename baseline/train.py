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

    def __init__(self, label: str) -> None:
        '''
        Classifier that assigns a weight to each label (and computes accuracy)
        '''
        self.label = label

        # Assign a weight to every tuple
        for label in learnFeatures.pmi:
            for weight in self.list_of_weights:
                classify_labels = dict(label = weight) 

    def learnFeatures(self, data):
        self.features = learnFeatures.learnFeatures(data)
        self.weights = [np.random.randint(-2,2) for i in range(len(self.features))]
        self.labels = sorted({feature.label for feature in self.features})
        print(f"The classifier learned {len(self.features)} features for {len(self.labels)} classes.")
    
    def classify(self, document: list[int]) -> str:
        """The classifier predicts the most probable label from its label set for a document given as a word vector.

        Args:
            document (list[int]): The document to be classified converted into a vector of 0's (word absent) and 1's (word included)

        Returns:
            str: The label with the highest probability for the given document
        """
        p = dict()
        for label in self.labels:
            p = sum()# TODO ?


    def train(self, data: list[tuple[tuple[int], str]]):
        ''' 
        Method that trains the model via multivariable linear optimization
        '''

        total_iterations = 100 
        n = 1
        old_lambda = self.weights
        new_lambda = list()
        for n in range(total_iterations): 
            if n != 1:
                old_lambda = new_lambda
            for i in self.features:
                new_lambda.append(old_lambda[i] - partial_derivative(old_lambda[i])
            n += 1
        residual = [x1 - x2 for (x1, x2) in zip(new_lambda, old_lambda)]
  
    def partial_derivative(self, lambda_i):

        # calculate first summand 'derivative_A'
        derivative_A = 0                         
        features(i) # How to refer to f? Input is old_lambda[i], we need feature[i]

            if y == self.label and x == self.document:
                derivative_A + 1

        # calculate second summmand 'derivative_B'
        derivative_B = 0                        
        for label in self.labels:
            for instance in instances:
                new_feature = dict(label = instance)
                new_features = ...
        for new_feature in new_features:
            if y == self.label and x == self.document:
                # probability pλ(y|x)
                # = exp(sum(weights_for_y_given_x)) / exp(sum(weights_for_y'_given_x))
                probability = np.exp(self.weight * derivative_A/ self.weight * x == document)
                derivative_B += probability

        #calculate derivative
        derivative = derivative_A - derivative_B
        return derivative  