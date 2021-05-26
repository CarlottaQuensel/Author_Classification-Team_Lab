# # -*- coding: utf-8 -*-

class Evaluation():
    """Evaluation for a trained multi-class single-label classifier.

    Methods:
    recall - compute the recall for every or a specific label
    precision - compute the precision for every or a specific label
    f1_score - compute the balanced f-score (alpha=0.5) from recall and precision
    confusion_matrix - erroneous classification as contingency table between gold and predicted labels
    """

    def __init__(self, gold_labels: list(str), predicted_labels: list(str)):
        """Initialize evaluation with the gold labels and the predictions to be evaluated.

        Args:
            gold_labels (list(str)): 
                    list of the gold classification labels used to evaluate the classifier
            predicted_labels (list(str)): 
                    list of the labels predicted by the classifier
                    the label is the one with the maximum probability
        """
        # The gold and predicted labels should be for the same documents, thus be the same number and order
        if len(gold_labels) != len(predicted_labels):
            raise IndexError("Unequal amount of labelled instances for gold and predicted labels. Use set_predictions to set again.")
        
        # Set the labels' numbering for consistent error counting and confusion matrix
        self.labels = sorted(set(gold_labels).union(set(predicted_labels)))
        # Set the gold and predicted labels as class features
        self.gold = gold_labels
        self.predicted = predicted_labels
        # Build the confusion matrix needed for the error calculation
        self.confusionMatrix = self.setConfusionMatrix()
    
    def set_predictions(self, gold_labels, predicted_labels):
        """Set gold and predicted labels for new data.

        Args:
            gold_labels (list(str)): 
                    list of the gold classification labels used to evaluate the classifier
            predicted_labels (liststr)): 
                    list of the labels predicted by the classifier
        """
        if len(gold_labels) != len(predicted_labels):
            raise IndexError("Unequal amount of labelled instances for gold and predicted labels. Use set_predictions to set again.")
        # If precision, recall and f-score were computed once, reset them as they don't match the current labels anymore
        self.precision_scores = list()
        self.recall_scores = list()
        self.f_scores = dict()

        # Set or reset the labels' numbering for consistent error counting and confusion matrix
        self.labels = sorted(set(gold_labels).union(set(predicted_labels)))
        # Set or overwrite the gold and predicted labels as class features
        self.gold = gold_labels
        self.predicted = predicted_labels

        # Build the confusion matrix needed for the error calculation
        self.setConfusionMatrix()

    
    def setConfusionMatrix(self, gold_labels=None, predicted_labels=None):
        """Generate confusion matrix as a contingency table between predicted and gold labels.

        Args:
            gold_output (list(list(int)), optional): 
                    List of the classifications for all texts in the form of 1 for the true label and 0 for all others. 
                    Defaults to None, then the set class property is used.
            predicted_output (list(list(float), optional): 
                    List of the classification probabilities for all texts, where the highest score is the predicted label. 
                    Defaults to None, then the class property is used.
        """
        # The CM can use the initial label set, an entirely new set or new predictions for the same gold labels
        if gold_labels and predicted_labels:
            self.set_predictions(gold_labels, predicted_labels)
        elif predicted_labels and self.gold:
            self.set_predictions(self.gold, predicted_labels)
        elif not self.gold:
            raise ValueError("The labels to compute a confusion matrix are missing, use set_predictions first")
        
        # initialize contingency table with 0 counts for all labels
        confusion = [[0 for i in range(len(self.labels))] for j in range(len(self.labels))]
        for i in range(len(self.gold)):
            true = self.labels[self.gold[i]]
            pred = self.labels[self.predicted[i]]
            confusion[true][pred] += 1
        return confusion


    def precision(self, label=None):
        """Compute precision for one or all labels as the ratio of predicted labels that are true

        Args:
            label (string, optional): If set, only the precision score for this label is computed. Defaults to None.
        """
        if label:
            # find label in the label dictionary and compute precision for the respective class
            try:
                label_index = self.labels.index(label)
            except ValueError:
                print(f"The given label {label} is not part of the label set")
                return None
            try:
                # Compute the precision as the ratio of correct classifications in all predicted instances
                prec =  self.confusionMatrix[label_index][label_index] / sum([self.confusionMatrix[j][label_index] for j in range(len(self.confusionMatrix))])
                print(f"Precision of {label}: {prec}")
                return prec
            except ZeroDivisionError:
                # If no instances were assigned the label, all no predictions were wrong
                print(f"Precision of {label}: 1")
                return 1
        else:
            # List for all individual precision scores to eventually compute the average
            prec_scores = list()
            # Compute precision for every class individually as
            for i in range(len(self.confusionMatrix)):
                # correctly assigned label
                tp = self.confusionMatrix[i][i]
                fp = sum([self.confusionMatrix[j][i] for j in range(len(self.confusionMatrix)) if j != i])
                # divided by all instances that were assigned the label
                prec_scores.append(tp/(tp+fp))
            self.prec = prec_scores
            # Show the precision for each label and their average
            prec_print = "Precision:\n"
            p1 = ""
            p2 = ""
            for i in range(len(prec_scores)):
                p1 += f"{self.labels[i]} "
                filler = len(self.labels[i])+1
                p2 += f"{prec_scores[i]:<{filler}}"
            av = sum(prec_scores)/len(prec_scores)
            prec_print += p1 + "Average\n" + p2 + str(av)
            print(prec_print)
            self.precision_scores = prec_scores
            return prec_scores

    
    def recall(self, label=None):
        """Compute recall for one or all labels as the ratio of gold labels that are predicted

        Args:
            label (string, optional): If set, only the recall score for this label is computed. Defaults to None.
        """
        if label:
            # If the label is given, compute recall for this one class
            try:
                # Find the label in the list of labels from the evaluated data
                label_index = [list(self.labels.keys())[list(self.labels.values()).index(label)]]
            except ValueError:
                # The label does not correspond to any label in the data
                print(f"The given label {label} is not part of the label set")
                return None
            try:
                # Compute the recall as the ratio of gold instances of the label which were found by the classifier
                return self.confusionMatrix[label_index][label_index] / sum(self.confusionMatrix[label_index])
            except ZeroDivisionError:
                # If none of the gold instances were found by the classifier, the recall is 0
                return 0
        else:
            # List for all individual recall scores to eventually compute the average
            rec_scores = list()
            # Compute recall for every class individually as
            for i in range(len(self.confusionMatrix)):
                # correctly assigned label
                tp = self.confusionMatrix[i][i]
                fn = sum([self.confusionMatrix[i][j] for j in range(len(self.confusionMatrix)) if j != i])
                # divided by all instances that should be assigned the label
                rec_scores.append( tp / (tp+fn) )
            self.rec = rec_scores
            # Show the recall for each label, their average
            rec_print = "Precision:\n"
            r1 = ""
            r2 = ""
            for i in range(len(rec_scores)):
                r1 += f"{self.labels[i]} "
                filler = len(self.labels[i])+1
                r2 += f"{rec_scores[i]:<{filler}}"
            av = sum(rec_scores)/len(rec_scores)
            rec_print += r1 + "Average\n" + r2 + str(av)
            print(rec_print)
            self.recall_scores = rec_scores
            return rec_scores
    

    def f_score(self, label: str = None, beta: int = 1) -> list[float]:
        """[summary]

        Args:
            label (str, optional): [description]. Defaults to None.
            beta (int, optional): [description]. Defaults to 1.

        Returns:
            list[float]: [description]
        """
        if label:
            # If the label is given, compute recall for this one class
            try:
                # Find the label in the list of labels from the evaluated data
                label_index = [list(self.labels.keys())[list(self.labels.values()).index(label)]]
            except ValueError:
                # The label does not correspond to any label in the data
                print(f"The given label {label} is not part of the label set")
                return None
            
            if len(self.f_scores) and beta in self.f_scores:
                f = self.f_scores[beta][label_index]
            else: 
                # Compute the recall as the ratio of gold instances of the label which were found by the classifier
                try:
                    f = (1+ beta*beta)*self.precision_scores[label_index]*self.recall_scores[label_index] / (beta*beta*self.precision_scores[label_index] + self.recall_scores[label_index])
                except ZeroDivisionError:
                    print(f"As both precision and recall for this label {label} are 0, no F_{beta}-score can be calculated")
                    return None
            print(f"F_{beta}('{label}') = {f}")
            return f
        
        else:
            # If there are already calculated f-scores with the given beta value, just print and return them
            if len(self.f_scores) and beta in self.f_scores:
                f_print = f"F_{beta} scores\n"
                f1 = ""
                f2 = ""
                for i, label in enumerate(self.labels):
                    f1 += label + " "
                    f = self.f_scores[beta][i]
                    f2 += f"{f:<{len(label)+1}}"
                f_print += f1 + "Average\n" + f2 + str(sum(self.f_scores[beta])/len(self.f_scores[beta]))
                print(f_print)
                return self.f_scores[beta]

            # Calculate precision and recall if not already done
            if not len(self.precision_scores):
                self.precision()
                self.recall()
            
            # List of F-scores for all labels
            f_scores = list()
            # Strings for showing the output
            f_print = f"F_{beta} scores:\n"
            f1 = ""
            f2 = ""
            # Calculate the f-score for each indivudual label
            for i, label in enumerate(self.labels):
                f1 += label + " "
                try:
                    f = (1+ beta*beta)*self.precision_scores[i]*self.recall_scores[i] / (beta*beta*self.precision_scores[i] + self.recall_scores[i])
                except ZeroDivisionError:
                    f = 0
                f2 += f"{f:<{len(label)+1}}"
                f_scores.append(f)
            # Print all individual f-scores and their average
            f_print += f1 + "Average\n" + f2 + str(sum(f_scores)/len(f_scores))
            print(f_print)
            # Add the f-scores with the current beta to the evaluation object and return them
            self.f_scores[beta] = f_scores
            return f_scores

    def showConfusionMatrix(self) -> dict[dict[int]]:
        if not self.confusionMatrix:
            print("There are no labels to compute a confusion matrix from, use set_predictions first")
            return None
        # _________________________________STRING FORMATING_________________________________ #
        # For the gold labels (rows), the name cell needs to be as long as the longest label
        max_label = max(list(map(len, self.labels)))

        # The string for the label columns is as long as all labels combined with a spacer in between each label.
        col_string = sum(list(map(len, self.labels)))+len(self.labels)-1

        # The first row only includes the axis description for the columns as the predicted labels
        cm_print = f"{' ':<{max_label+1}}{'predicted:':<{col_string}}|\n"

        # The column titles are the labels delimited by | as a vertical line
        col_titles = f"{' ':<{max_label}}"
        # The lines between rows are dashes with + where they meet a vertical line between columns
        hlines = f"{'-':{'-'}<{max_label}}+"
        vlines = ""
        for label in self.labels:
            col_titles += label + "|"
            hlines += f"{'-':{'-'}<{len(label)}}+"
            vlines += f"{' ':{len(label)}}|"
        # Adding the column titles, the first horizontal line and the axis description for the rows as the gold labels
        cm_print += col_titles + "\n"
        cm_print += hlines + "\n" + f"{'gold:':<{max_label}}|{vlines}\n"
        # _________________________________STRING FORMATING_________________________________ #

        # Each gold label has its own entry in a dictionary and its own row in the printed table
        cm = dict()
        for i, row_label in enumerate(self.labels):
            # Initializing the predictions for the current gold label
            cm[row_label] = dict()
            # Adding the row title to the output string
            row_string = f"{row_label:<{max_label}}|"
            for j, col_label in enumerate(self.labels):
                # Adding the number of times, the gold row label was predicted as the current column label
                cm[row_label][col_label] = self.confusionMatrix[i][j]
                # Adding the number as a cell of the printed table
                row_string += f"{self.confusionMatrix[i][j]:<{col_label}}|"
            # Adding the whole formated row and a horizontal line to the printed table
            cm_print += row_string + "\n" + hlines + "\n"
        
        # After the whole matrix is traversed, it is printed in a readable format and returned as an interpretable dictionary with the respective labels
        print(cm_print)
        return cm


