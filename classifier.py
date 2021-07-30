# -*- coding: utf-8 -*-
from features import *
from document import Poem
import numpy as np


class MaxEnt():

    # Initialize a list of features and corresponding weights
    weights = list()
    features = list()

    def __init__(self, data: list[tuple[Poem, str]] = None, class_features: int = None) -> None:
        """
        Initialize a Maximum Entropy classifier and learn the features and weights if data is given.
        Author: Katrin Schmidt

        Args:
            data (list[tuple[Poem, str]], optional):
                Dataset as a list of Poem-author pairs. Defaults to None.
            class_features (int, optional):
                Number of max learned features per class. Defaults to None.
        """
        if data:
            if class_features:
                self.learnFeatures(data, class_features)
            else:
                self.learnFeatures(data)

    def learnFeatures(self, data: list[tuple[Poem, str]], bow_features: int = 30, verse_features: bool = True, rhyme_features: int = 0, vocabulary: list[str] = None, trace: bool = False) -> None:
        """
        Initialize F and λi for the classifier by learning the most informative features for every author.
        
        The features consist of an author, a form and a document property, e.g.
        label="Shakespeare" and property="aabb" (form="rhyme_scheme")
        Author: Carlotta Quensel (see module features.py)

        Args:
            data (list[tuple[Poem, str]]):
                Dataset consisting of a list of poem author pairs,
                where the poems have a word vector, verse count and rhyme scheme
            bow_features (int, optional):
                The number of bag of word features learned per author, defaults to 30.
            verse_features (bool, optional):
                Determines if the classifier learns verse counts or not, defaults to False.
            rhyme_features (int, optional):
                Determines how many (if any) rhyme scheme features are learned per author,
                defaults to 0.
            vocabulary (dict[str], optional):
                The assignment of words to word vector indices used 
                to show learned features if the user set trace=True
            trace (bool, optional):
                Given the vocabulary, show a selection of learned features
        """
        if vocabulary:
            self.vocabulary = vocabulary

        # Learn features (Fi) using PMI (explained in learnFeatures.py)
        self.features = learnFeatures(
            data, bow_features, verse_features, rhyme_features)

        # Initialize a random weight for each learned feature
        self.weights = [1 for i in range(len(self.features))]

        # Save all authors of the training data to simplify classification
        self.labels = sorted({feature.label for feature in self.features})

        # Show a selection of learned features if the user tracks the training process
        print(
            f"The classifier learned {len(self.features)} features for {len(self.labels)} classes.")
        if trace and vocabulary:
            for i in range(0, len(self.features)):
                if i % 30 in {0, 1, 2, 3, 29}:
                    if self.features[i].form == "bow":
                        print(
                            f"{i} - author: {self.features[i].label}, poem contains {list(self.vocabulary)[self.features[i].property]}")
                    elif self.features[i].form == "verse":
                        print(
                            f"{i} - author: {self.features[i].label}, poem has {self.features[i].property[0]+1} - {self.features[i].property[1]} verses")
                    elif self.features[i].form == "rhyme_scheme":
                        print(
                            f"{i} - author: {self.features[i].label}, poem has a {self.features[i].property} rhyme scheme")

    def classify(self, document: list[int], in_training: str = False, weights: list[int] = None):
        """
        Predicts the most probable of all learned authors for a poem or returns all probabilities in training. 

        The Maximum Entropy classifier works by applying a binary function to
        a poem label pair to switch a weight on or off depending on a feature of 
        the poem and the author. The score of an author is determined as the 
        exponential of the sum of all switched on weights normalized by the sum of
        all authors' scores: p(a|poem) = exp(Σᵢ wᵢ·fᵢ(a,poem)) / Σ(a') exp(Σᵢ wᵢ·fᵢ(a',poem)).
        Author: Carlotta Quensel

        Args:
            document (list[int]):
                The document to be classified converted into
                a vector of 0's (word absent) and 1's (word included)
            in_training (bool, optional):
                When computing the derivative in training, the classifier uses
                the probabilities itself and not the label. Defaults to False.
            weights (list[int], optional):
                If testing the accuracy of a specific weight set, custom weights
                can be used. Defaults to None.

        Returns:
            None: If the custom weights don't match the classifier's number of
            functions, it cannot compute any probabilities
            str: Outside of the training, the label with the highest probability
            for the given document is returned
            dict[str]: When training, the probability of all labels is returned
            for the derivative calculation

        """
        # Set the feature weights to the classifier weights as default
        if not weights:
            weights = self.weights
        elif len(weights) != len(self.weights):
            print(
                f"The classifier needs exactly one weight per function. You used {len(weights)} weights for {len(self.features)} functions.")
            return None

        # Add up Fi(poem, author)*λi for all features and weights for an auhtor's score given a poem
        numerator = dict()
        for label in self.labels:
            numerator[label] = np.exp(sum(
                [weights[i]*self.features[i].apply(label, document) for i in range(len(self.features))]))

        # Normalize an author's score by the sum of all author's
        # Same denominator for all authors = exp(Σauthors Σi Fi(poem, author)*λi 
        denominator = sum(numerator.values())

        # The probability of a label then just divides the numerator by the denominator
        p = dict()
        for label in self.labels:
            p[label] = numerator[label] / denominator

        # Return the most probable author, or in training return all probabilities
        if in_training:
            return p
        else:
            return max(p.keys(), key=lambda x: p[x])

    def accuracy(self, data: list[tuple[Poem, str]], weights: list[int] = None) -> float:
        """
        Compute the basic accuracy of the classifier with a specific weight set as
        the percentage of correct predictions from the overall number of predictions
        Author: Carlotta Quensel

        Args:
            data (list[tuple[Poem, str]]):
                List of datapoints as pairs of a Poem and an author string
            weights (list[int], optional):
                Custom weights to compare with the classifier's current weight set.
                Defaults to None.

        Returns:
            float: The accuracy as the ratio of correct predictions from all predictions
        """
        if not weights:
            weights = self.weights
        # Count true positives
        tp = 0
        for doc, label in data:
            prediction = self.classify(document=doc, weights=weights)
            if label == prediction:
                tp += 1
        # Accuracy = TP / TP+FN (in this multi-label setup, FP and TN are not counted)
        return tp/len(data)

    def train(self, data: list[tuple[Poem, str]], min_improvement: float = 0.001, trace: bool = False):
        '''
        Method that trains the model via multivariable linear optimization.
        Since the optimization for the lambda vector needs to happen simultaneous,
        the iterations stop after it counts 100 (instead of a specific value).
        Author: Katrin Schmidt (original code, for detailed changes see 
                midterm submission)
                Carlotta Quensel (trace, loss and accuracy, switch from 
                calculating individual partial derivatives to gradient)

        Args:
            data (list[tuple[Poem, str]]):
                Dataset as a list of Poem-author pairs.
            min_improvement (float, optional):
                The minimum accuracy improvement needed for a new iteration of
                optimization. Defaults to 0.0001
            trace (boolean, optional): Switch on to track the development of
                the accuracy during the iterative process.
        '''

        # Track accuracy and loss during training
        old_accuracy = 0
        loss = list()
        new_accuracy = self.accuracy(data)
        # Show the initial accuracy if wanted by the user
        if trace:
            print(f"Accuracy with random weights: {new_accuracy}.")
            acc = [new_accuracy]
        new_lambda = list()

        # Keep track of the number of optimization steps
        i = 1

        # Stop optimization after at least ten iteration steps or when the 
        # accuracy converges (default improvement threshold: 0.1%)
        while new_accuracy - old_accuracy >= min_improvement or i < 11:

            # Don't reassign the new weights in the first step
            if len(new_lambda):
                self.weights = new_lambda

            # Compute the gradient of F
            gradient = self.partial_derivative(data)
            # Optimize the weights through stochastic gradient ascent
            new_lambda = [self.weights[i] + gradient[i] for i in range(len(self.features))]
            # Track the loss as the gradient's sum
            new_loss = abs(sum(gradient))
            # Update the accuracy with the optimized weights
            old_accuracy = new_accuracy
            new_accuracy = self.accuracy(data, new_lambda)

            # Keep track of the accuracy and loss optimization for each iteration
            if trace:
                print(
                    f"iteration {i:2} : accuracy {new_accuracy}\n{' ':16}loss {new_loss}")
                acc.append(new_accuracy)
                loss.append(new_loss)
            i += 1

        # Show and return the training progress if the user wants to track it
        if trace:
            print(
                f"The training consisted of {i-1} optimization steps in which the accuracy changed from {acc[0]} to {acc[-1]} and the error changed from {loss[0]} to {loss[-1]}.")
            return acc, loss

    def partial_derivative(self, data: list[tuple[Poem, str]]) -> list[float]:
        '''
        Computes the gradient as a vector of partial derivative of the objective function
        F_λi(x,y) with regard to λi.
        Author: Katrin Schmidt (original code for individual partial derivatives λi, 
                see midterm submission), 
                Carlotta Quensel (for time optimization with new Poem class: conversion 
                into gradient from all λi, list abstractions, getting p_λi(label,document)
                for all documents/labels at once instead of for each λi)

        Args:
            data (list[tuple[Poem,str]]):
                Dataset as a list of Poem-author pairs to apply
                the partial dervative function to for each weight λi
        Returns:
            list[float]: Gradient comprised of partial derivatives of the function F
        '''
        # Calculate dF/dλi as dA/dλi - dB/dλi
        # dA/dλi = Σdata F_λi(poem,author) 
        #        number of poem author training pairs correctly recognized by F
        derivative_A = [sum([self.features[lambda_i].apply(
            gold_label, document) for (document, gold_label) in data]) for lambda_i in range(len(self.weights))]

        # dB/dλi = Σdata Σauthors F_λi(poem,author) 
        #          add up the probabilities of all poem author pairs 
        #          recognized by F from the training poems
        derivative_B = []
        # Probabilities for every author sorted by the poems
        p_prime = [self.classify(document, in_training=True)
                   for document, gold_label in data]
        # For each weight, add up the probabilities of each poem author combination that F applies to
        for lambda_i in range(len(self.weights)):
            derivative_B.append(sum([sum([self.features[lambda_i].apply(prime_label, document)*p_prime[i][prime_label]
                                for prime_label in p_prime[i]]) for i, (document, gold_label) in enumerate(data)]))

        # Calculate gradient of F
        return [derivative_A[i]-derivative_B[i] for i in range(len(self.weights))]
