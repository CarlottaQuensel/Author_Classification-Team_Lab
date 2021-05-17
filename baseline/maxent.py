# -*- coding: utf-8 -*-
# TODO: training function
# TODO: prediction/classification function
from baseline.features import Feature


class MaximumEntropy():
    features = list()
    weights = list(int())

    def classify(self, document_vector: list[list[int]]) -> str:
        label_prob = float()
        for label in self.labels:
            for i, feature in enumerate(self.features):
                # what now?