# -*- coding: utf-8 -*-
from features import *
import numpy as np


class MaxEnt():
    '''
    Author: Katrin Schmidt
    1. Initialize random weights between -10 and +10 (= lambda)
    2. Calculate accuracy of application of these lambdas
    3. Improve accuracy by calculation the derivation and adjusting the lambdas
    4. Return best lambdas for each feature
    '''

    # Initialize a list of features and corresponding weights
    weights = list()
    features = list()

    def __init__(self, data: list[tuple[tuple[int], str]]=None, class_features: int=None) -> None:
        """
        Author: Katrin Schmidt
        Initializing an instance of a Maximum Entropy classifier, if data is
        already given, then also learning class features
        Args:
            data (list[tuple[tuple[int], str]], optional):
                Dataset as a list of document vector-label pairs. Defaults to None.
            class_features (int, optional):
                Number of max learned features per class. Defaults to None.
        """
        if data:
            if class_features:
                self.learnFeatures(data, class_features)
            else:
                self.learnFeatures(data)

    def learnFeatures(self, data: list[tuple[tuple[int], str]], class_features: int=50, vocabulary: list[str] = None) -> None:
        """
        Author: Carlotta Quensel (see module features.py)
        Compute the best features for the classifier based on pmi between
        classes and document features in a dataset and save them in a list with
        a matching list of (untrained) weights. The features are saved as
        functions for the Max Ent classification and have an associated label
        and a document property as a number, e.g.
        -> label="Shakespeare" and property=45 (index of "Thou", is in the document).

        Args:
            data (list[tuple[tuple[int], str]]):
                Dataset consisting of a list of document-label pairs,
                where the documents are word vectors
            class_features (int):
                The maximum number of features learned per class, default 50
            vocabulary (dict[str]):
                The assignment of words to word vector indices used
                in the given data set
        """
        if vocabulary:
            self.vocabulary = vocabulary

        # Learning the feature-functions uses PMI and is explained more
        # thoroughly in learnFeatures.py
        self.features = learnFeatures(data, class_features, vocab=vocabulary)

        # Each function has a corresponding weight, so the same number of
        # weights are randomly initialized
        self.weights = [np.random.randint(-10, 10) for i in range(len(self.features))]

        # The classifier also has all labels of the training data saved
        # to simplify classification
        self.labels = sorted({feature.label for feature in self.features})

        # After learning the features, information about the number of
        # features and labels is shown
        print(f"The classifier learned {len(self.features)} features for {len(self.labels)} classes.")

    def classify(self, document: list[int], in_training: str=False, weights: list[int]=None):
        """
        Author: Carlotta Quensel
        The classifier predicts the most probable label from its label set for a
        document given as a word vector. The Maximum Entropy classifier works
        with a function of a document and label, returning either 1 if if applies
        or 0 if it doesn't apply to the document-label pair and the function's
        correspinding weight. The probability of a label y given a document is
        the exponential function of the sum of all weights with applied functions
        divided (normalized) by the sum of this exponential for all possible labels:
        p(y|doc) = exp(????? w?????f???(y,doc)) / ??(y') exp(????? w?????f???(y',doc)).
        The classifier computes probabilities for all labels and returns the
        highest scoring label or, in training, the score itself

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
        # The default weights to compute the probabilities are the classifier's own weights
        if not weights:
            weights = self.weights
        elif len(weights) != len(self.weights):
            print(f"The classifier needs exactly one weight per function. You used {len(weights)} weights for {len(self.features)} functions.")
            return None

        # The numerator of the Max Ent-probability is the exponential function
        # of every weight*function with the current label and given document
        numerator = dict()
        for label in self.labels:
            numerator[label] = np.exp(sum([weights[i]*self.features[i].apply(label, document) for i in range(len(self.features))]))

        # As the denominator is the sum of the exponential for all labels, it
        # only depends on the document and is the same for every label
        denominator = sum(numerator.values())

        # The probability of a label then just divides the numerator by the denominator
        p = dict()
        for label in self.labels:
            p[label] = numerator[label] / denominator

        # The classifier either returns the most probable label or in training
        # returns the label's probability
        if in_training:
            return p[in_training]
        else:
            return max(p.keys(), key=lambda x: p[x])

    def accuracy(self, data: list[tuple[tuple[int], str]], weights: list[int]=None) -> float:
        """
        Author: Carlotta Quensel
        Compute the basic accuracy of the classifier with a specific weight set as
        the percentage of correct predictions from the overall number of predictions

        Args:
            data (list[tuple[tuple[int], str]]):
                List of datapoints as pairs of a document vector and a label string
            weights (list[int], optional):
                Custom weights to compare with the classifier's current weight set.
                Defaults to None.

        Returns:
            float: The accuracy as the ratio of correct predictions from all predictions
        """
        if not weights:
            weights = self.weights
        tp = 0
        for doc, label in data:
            prediction = self.classify(document=doc, weights=weights)
            if label == prediction:
                tp += 1
        return tp/len(data)

    def train(self, data: list[tuple[tuple[int], str]], min_improvement: float = 0.001, trace: bool = False):
        '''
        Author: Katrin Schmidt (main),
                Carlotta Quensel (trace, loss and accuracy)
        Method that trains the model via multivariable linear optimization.
        Since the optimization for the lambda vector needs to happen simultaneous,
        the iterations stop after it counts 100 (instead of a specific value).

        Args:
            data (list[tuple[tuple[int], str]]):
                Dataset as a list of document vector-label pairs.
            min_improvement (float, optional):
                The minimum accuracy improvement needed for a new iteration of
                optimization. Defaults to 0.0001
            trace (boolean, optional): Switch on to track the development of
                the accuracy during the iterative process.
        '''

        # First compute old accuracy with random weights
        # and keep track of the overall loss (sum of the gradient)
        old_accuracy = 0
        loss = list()
        new_accuracy = self.accuracy(data)
        if trace:
            print(f"Accuracy with random weights: {new_accuracy}.")
            acc = [new_accuracy]
        new_lambda = list()
        
        # Set control variable i to 1 to keep track of the optimization steps
        i = 1

        # Optimize the weights until the improvement drops below a
        # threshold (currently at least 0.1% improvement in accuracy)
        while new_accuracy - old_accuracy >= min_improvement:

            # Don't reassign the new weights in the first step
            if len(new_lambda):
                self.weights = new_lambda

            # Iterate over features and compute the partial_derivative
            # and save the updated weights temporarily
            new_lambda = [self.weights[i] + self.partial_derivative(data, lambda_i=i) for i in range(len(self.features))]

            # Update accuracy and derivative or loss to check improvement
            # (acc -> 1, loss -> 0)
            new_loss = abs(sum([x2-x1 for (x1, x2) in zip(new_lambda, self.weights)]))
            old_accuracy = new_accuracy
            new_accuracy = self.accuracy(data, new_lambda)

            # Track the training progress by printing accuracy and loss
            # for each optimization step
            if trace:
                print(f"iteration {i:2} : accuracy {new_accuracy}\n{' ':16}loss {new_loss}")
                acc.append(new_accuracy)
                loss.append(new_loss)
            i += 1

        if trace:
            print(f"The training consisted of {i-1} optimization steps in which the accuracy changed from {acc[0]} to {acc[-1]} and the error changed from {loss[0]} to {loss[-1]}.")
            # If the user wants to track the training process, the accuracy and
            # loss scores are returned to potentially plot the improvement
            return acc, loss

    def partial_derivative(self, data: list[tuple[tuple[int], str]], lambda_i: int) -> list[float]:
        '''
        Author: Katrin Schmidt
        Method that computes the partial derivatives of the objective function F
        by substracting the derivative of A from the derivative of B with regard
        to ??i.

        Args:
            data (list[tuple[tuple[int],str]]):
                Dataset as a list of document vector-label pairs to apply
                the partial dervative function to for each weight ??i
            lambda_i (int):
                Index of the weight with regard to which the derivative
                is currently calculated
        Returns:
            list[float]: Gradient comprised of partial derivatives of the function F
        '''

        # Set derivative of A and B to zero
        derivative_A = 0
        derivative_B = 0

        for (document, gold_label) in data:

            # Calculate first summand by counting the number of
            # correctly recognized document-label pairs
            derivative_A += self.features[lambda_i].apply(gold_label, document)

            # Calculate second summmand as the probability of all
            # label-doc pairs that could be recognized
            for prime_label in self.labels:

                # Feature is either 0 or 1 -> can be replaced with an if-query
                # to minimize the number of classifications during training
                # that would be multiplied with 0:
                if self.features[lambda_i].apply(prime_label, document):

                    # Probability p??(y|x) = exp(sum(weights_for_y_given_x)) /
                    #                       exp(sum(weights_for_y'_given_x))
                    derivative_B += self.classify(document, in_training=prime_label)

        # Calculate derivative of F
        return derivative_A-derivative_B
