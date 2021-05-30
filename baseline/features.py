# -*- coding: utf-8 -*-
# TODO: make comment for class description

class Feature():
    label = str()
    property = index, included = int(), bool()


    def __init__(self, label: str, doc_property: tuple[int, bool]) -> None:
        """Initialization of a maximum entropy feature.
        A MaxEnt feature is a binary function that where a document either has a property related to a label or not.

        Args:
            label (str): The label whose weight the function switches on or off.
            doc_property (tuple): 
                index (int): The index of the feature's word/property in the document vector.
                inclusion (bool): Stating if the feature calls for inclusion or absence of a property in the documents.
        """
        self.label = label
        index = doc_property[0]
        included = doc_property[1]
        self.index = index
        self.included = included


    def apply(self, current_label: str, doc: list[int]) -> int:
        """Compares a maximum entropy feature to a current document and label, to decide if the MaxEnt classifier
        will apply the feature and thus include a weight for this instance (1) or not (0).

        Args:
            current_label (str): The label whose probability the classifier should compute.
            doc (list[int]): The document instance currently being classified, as a bag-of-words feature vector.

        Returns:
            1: If the feature can be applied to the document and the label matches with the probability currently computed.
            0: If the document doesn't match the property the feature needs or another label is currently considered.
        """
        # The feature calls for the document to include a word or property.
        if self.included:
            switch = (self.label == current_label and doc[self.property])
        # The feature calls for the document not to include a word or property
        else:
            switch = (self.label == current_label and not doc[self.property])
        # The boolean on/off-switch is converted to a number to be multiplied with the weight
        return int(switch)