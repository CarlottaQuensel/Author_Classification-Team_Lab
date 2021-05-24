# -*- coding: utf-8 -*-
from baseline.learnFeatures import learnFeatures
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
    list_of_weights = []
    random_weight = np.random.randint(-2,2)
    weights = list_of_weights.append(random_weight * len(learnFeatures.labels))

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

    
    def train(self):
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
                for feature in features:
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