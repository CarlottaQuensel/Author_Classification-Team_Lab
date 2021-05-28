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
    '''list_of_weights = []
    random_weight = np.random.randint(-2,2)
    weights = list_of_weights.append(random_weight * len(learnFeatures.labels))'''

    def __init__(self, label: str) -> None:
        '''
        Classifier that assigns a weight to each label and computes accuracy 
        '''
        self.label = label

        # Assign a weight to every tuple
        for label in learnFeatures.pmi:
            for weight in self.list_of_weights:
                classify_labels = dict(label = weight) 
        old_accuracy = (self.labels/learnFeatures.doc_number)

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
        Method that calls accuracy and compares to previous time step
        '''
        
        new_accuracy = 
        delta = new_accuracy - self.old_accuracy

        # Adjust the weight until convergence
        while delta > new_accuracy - self.old_accuracy:

            # Compute it for all instances
            for instance in instances:

                # Compute the derivative of A
                # δA/δλ = sum( fi(y,x) )
                derivative_of_A = 0                         
                for feature in self.features:
                    if y == self.label and x == document:
                        derivative_of_A + 1

                # Compute the derivative of B
                # δB/δλ = p(y'|x) fi(y',x)
                derivative_of_B = 0                        
                for label in self.labels:
                    for instance in instances:
                        new_feature = dict(label = instance)
                        for new_feature in new_features:
                            if y == self.label and x == document:
                                probability = np.exp(weight * derivative_of_A/ weight * x == document)
                                derivative_of_B += probability

            # Compute the derivative of F                    
            derivative_of_F = derivative_of_A - derivative_of_B
            new_lambda = self.old_accuracy - derivative_of_F


            # new_features / classify_labels ?
            # where to initialize?