# -*- coding: utf-8 -*-


class Evaluation():
    """Evaluation for a multi-class single-label classifier.

    Methods:
    recall  - compute the recall for every or a specific label
    precision - compute the precision for every or a specific label
    f1_score - compute the balanced f-score (alpha=0.5) from recall and precision
    showConfusionMatrix - erroneous classification as contingency table between
                          gold and predicted labels
    fullEval - compute and show all evaluation measures and the confusion matrix
               for any given data set
    """
    precision_scores = dict()
    recall_scores = dict()
    f_scores = dict()

    def __init__(self, gold_labels: list[str], predicted_labels: list[str]):
        """Initialize the evaluation by setting the label set.
        Author: Katrin Schmidt

        Args:
            gold_labels (list(str)):
                    List of the gold classification labels used
                    to evaluate the classifier
            predicted_labels (list(str)):
                    List of the labels predicted by the classifier
                    the label is the one with the maximum probability
        """

        # Mapping gold and predicted labels to each other is not possible
        # ff the two label sets don't have the same size
        if len(gold_labels) != len(predicted_labels):
            raise IndexError("Unequal amount of labelled instances for gold and predicted labels.\
                Use setPredictions to set again.")

        # Sort all labels in both gold and predictions as a reference for
        # the confusion matrix and the evaluation measures
        self.labels = sorted(set(gold_labels).union(set(predicted_labels)))

        # Set the gold and predicted labels as class features
        self.gold = gold_labels
        self.predicted = predicted_labels

        # Build the confusion matrix needed for the error calculation
        self.confusionMatrix = self.setConfusionMatrix()

    def setPredictions(self, predicted_labels, gold_labels=None):
        """Reset the evaluated label sets if the predictions have changed
        Author: Carlotta Quensel

        Args:
            gold_labels (list(str), optional):
                    Correct labels against which the predictions are evaluated
                    If not given, it defaults to the already set gold labels
            predicted_labels (list(str)):
                    Predictions of a classifier to be evaluated
        """
        if not gold_labels:
            gold_labels = self.gold
        if len(gold_labels) != len(predicted_labels):
            raise IndexError(
                "Unequal amount of labelled instances for gold and predicted labels. \
                Use setPredictions to set again.")
        # Reset all evaluation scores from the old labels
        self.precision_scores = list()
        self.recall_scores = list()
        self.f_scores = dict()

        # Reset the label numbering for consistent error counting and confusion matrix
        self.labels = sorted(set(gold_labels).union(set(predicted_labels)))
        # Reset the gold and predicted labels as class features
        self.gold = gold_labels
        self.predicted = predicted_labels

        # Build the confusion matrix needed for the error calculation
        self.confusionMatrix = self.setConfusionMatrix()

    def setConfusionMatrix(self, gold_labels=None, predicted_labels=None):
        """Generate a confusion matrix as a contingency table between 
        predicted and gold labels.
        Author: Carlotta Quensel

        Args:
            gold_output (list(list(int)), optional):
                    List of the classifications for all texts in the form of 1
                    for the true label and 0 for all others.
                    Defaults to None, then the set class property is used.
            predicted_output (list(list(float), optional):
                    List of the classification probabilities for all texts,
                    where the highest score is the predicted label.
                    Defaults to None, then the class property is used.
        
        Returns:
            list[list[int]]: The contingency table with gold labels as rows
                and predicted labels as columns
        """

        # Either use newly given labels or the initial label set
        if gold_labels and predicted_labels:
            self.set_predictions(gold_labels, predicted_labels)
        elif predicted_labels and self.gold:
            self.set_predictions(self.gold, predicted_labels)
        elif not self.gold:
            raise ValueError(
                "The labels to compute a confusion matrix are missing, \
                use setPredictions first")

        # Initialize contingency table with 0 counts for all labels
        confusion = [[0 for i in range(len(self.labels))]
                     for j in range(len(self.labels))]
        # Count each prediction by incrementing the cell corresponding
        # to the row of the gold label and column of the predicted label
        for i in range(len(self.gold)):
            true = self.labels.index(self.gold[i])
            pred = self.labels.index(self.predicted[i])
            confusion[true][pred] += 1

        # Return the confusion matrix to be used to calculate precision and recall
        return confusion

    def precision(self, label: str = None, full_eva: bool = False) -> list[float]:
        """Compute and return the precision for one or all labels as 
        the ratio of predicted labels that are true.
        Author: Katrin Schmidt
                Carlotta Quensel (Error handling, average and formatted output)

        Args:
            label (string, optional):
                If set, only the precision score for this label
                is computed and shown. Defaults to None.
            full_eva (boolean, optional):
                If the precision is computed as part of a full evaluation,
                the results are shown in the respective function -> not immediately.

        Returns:
            list[float]:
                The precision scores for each label in the same order
                as the class' label set
        """

        # If the user determines a label, return only its precision
        if label:
            # Find the label's index in the label dictionary
            try:
                label_index = self.labels.index(label)
            except ValueError:
                print(f"The given label {label} is not part of the label set")
                return None

            # Compute precision as the ratio of correct 
            # classifications in all predicted instances
            try:
                prec = self.confusionMatrix[label_index][label_index] / sum(
                    [self.confusionMatrix[j][label_index] for j in range(len(self.confusionMatrix))])
                print(f"Precision of {label}: {round(prec,4)}")
                return prec

            # Return 1 if the label was never predicted, as
            # no prediction with this label was wrong
            except ZeroDivisionError:
                print(f"Precision of {label}: 1")
                return 1.0

        # Without determining a specific label, calculate the precision
        # of all labels and their average
        else:
            # List for all individual precision scores to eventually compute the average
            prec_scores = list()

            # Compute precision for every class individually as:
            for i in range(len(self.confusionMatrix)):

                # the correctly assigned label
                tp = self.confusionMatrix[i][i]
                fp = sum([self.confusionMatrix[j][i]
                         for j in range(len(self.confusionMatrix)) if j != i])

                # divided by all instances that were assigned the label
                try:
                    prec_scores.append(tp/(tp+fp))
                except ZeroDivisionError:
                    # Set the precision to 1 for never predicted labels
                    prec_scores.append(1.0)
            self.prec = prec_scores

            # In performing a full evaluation, return the results to the 
            # full evaluation function to be shown together with the
            # other evaluation scores.
            if full_eva:
                self.precision_scores = prec_scores
                return prec_scores

            # Outside of a full evaluation, output a formatted table of the 
            # precision for each label and their micro average
            prec_print = 20*"_" + "\nPrecision:\n"
            p1 = ""
            p2 = ""

            for i in range(len(prec_scores)):
                p1 += f"{self.labels[i]} "
                filler = len(self.labels[i])+1
                p2 += f"{round(prec_scores[i], 4) : <{filler}}"

            # Average
            av = sum(prec_scores)/len(prec_scores)
            prec_print += p1 + "Average\n" + p2 + str(round(av, 4))
            print(prec_print)

            # Return precision scores
            self.precision_scores = prec_scores
            return prec_scores

    def recall(self, label=None, full_eva: bool = False) -> list[float]:
        """Compute the recall of a classifier for one or all labels as
        the ratio of correct predictions from all instances of a label.
        Author: Katrin Schmidt
                Carlotta Quensel (Error handling, average and formatted output)

        Args:
            label (string, optional):
                If set, only the recall score for this label
                is computed and shown. Defaults to None.
            full_eva (bool, optional):
                If the recall is computed as part of a full evaluation,
                the scores are shown with all others and not immediately.
                Defaults to False.

        Returns:
            list[float]: Recall scores for each label in the same order
            as the class' label set
        """

        # If the label is given, compute recall for this one class
        if label:
            # Find the label's index from the label dictionary
            try:
                label_index = [list(self.labels.keys())[
                    list(self.labels.values()).index(label)]]
            except ValueError:
                print(f"The given label {label} is not part of the label set")
                return None

            # Compute recall as the ratio of gold instances of the label
            # found by the classifier
            try:
                return self.confusionMatrix[label_index][label_index] / sum(self.confusionMatrix[label_index])

            # If none of the gold instances were found by the classifier,
            # the recall is 0
            except ZeroDivisionError:
                return 0.0

        # Without a specific label, list all labels' scores and their average
        else:
            rec_scores = list()

            # Compute recall for every class individually as:
            for i in range(len(self.confusionMatrix)):

                # correctly assigned label
                tp = self.confusionMatrix[i][i]
                fn = sum([self.confusionMatrix[i][j]
                         for j in range(len(self.confusionMatrix)) if j != i])

                # divided by all instances that should be assigned
                try:
                    rec_scores.append(tp / (tp+fn))
                except ZeroDivisionError:
                    rec_scores.append(0)
            self.rec = rec_scores

            # In performing a full evaluation, return the results to the 
            # full evaluation function to be shown together with the
            # other evaluation scores.
            if full_eva:
                self.recall_scores = rec_scores
                return rec_scores

            # Outside of a full evaluation, output a formatted table of the 
            # precision for each label and their micro average
            rec_print = 20*"_" + "\nRecall:\n"
            r1 = ""
            r2 = ""
            # Format individual scores
            for i in range(len(rec_scores)):
                r1 += f"{self.labels[i]} "
                filler = len(self.labels[i])+1
                r2 += f"{round(rec_scores[i],4):<{filler}}"
            # Add average to the results
            av = sum(rec_scores)/len(rec_scores)
            rec_print += r1 + "Average\n" + r2 + str(round(av, 4))
            print(rec_print)

            # Return recall scores
            self.recall_scores = rec_scores
            return rec_scores

    def f_score(self, label: str = None, beta: int = 1, full_eva: bool = False) -> list[float]:
        """
        Compute and show the f-score as a combination of precision and recall.
        Per default, their harmonic mean is computed.
        Author: Katrin Schmidt
                Carlotta Quensel (Error handling, average and formatted output)

        Args:
            label (str, optional):
                Only compute and show the score for a specific label.
                Defaults to None.
            beta (int, optional):
                The beta-value of the f-formula that scales the influence of the recall.
                For beta<1, the precision is weighted higher, for beta>1, the recall is weighted higher
                Defaults to 1 to compute the harmonic mean.
            full_eva (boolean, optional):
                If the f-score is computed as part of a full evaluation, the results are shown with this
                full evaluation and not immediately after computation. Defaults to False.

        Returns:
            list[float]: F-scores for each label in the same order as the class' label set.
        """

        # Given a label, compute recall for this one class
        if label:
            # Find the label's index in the label dictionary
            try:
                label_index = [list(self.labels.keys())[
                    list(self.labels.values()).index(label)]]
            except ValueError:
                print(f"The given label {label} is not part of the label set")
                return None

            # Return the f-score, if it was already computed with the current beta value
            if len(self.f_scores) and beta in self.f_scores:
                f = self.f_scores[beta][label_index]

            # Compute the f-score from the precision and recall score of the label
            else:
                try:
                    f = (1 + beta*beta)*self.precision_scores[label_index]*self.recall_scores[label_index] / (
                        beta*beta*self.precision_scores[label_index] + self.recall_scores[label_index])
                except ZeroDivisionError:
                    f = 0.0
            print(f"F_{beta}('{label}') = {f}")
            return f

        # Without a label given, calculate the f-score for all labels and 
        # their micro- and macro average
        else:
            # Use the f-score, if it was already computed with 
            # the current beta value
            if len(self.f_scores) and beta in self.f_scores:
                # In a full evaluation, return the scores to be
                # formatted later
                if full_eva:
                    return self.f_scores[beta]
                # Format the scores to show as output
                f_print = 20*"_" + f"\nF_{beta} scores\n"
                f1 = ""
                f2 = ""
                for i, label in enumerate(self.labels):
                    f1 += label + " "
                    f = self.f_scores[beta][i]
                    f2 += f"{round(f,4):<{len(label)+1}}"
                f_print += f1 + "Average\n" + f2 + \
                    str(round(
                        sum(self.f_scores[beta])/len(self.f_scores[beta]), 4))
                print(f_print)
                # Return the scores of all labels and the micro- and macro-f-score
                return self.f_scores[beta]

            # Calculate precision and recall if not already done
            if not len(self.precision_scores) or not len(self.recall_scores):
                self.precision()
                self.recall()

            # List of F-scores for all labels
            f_scores = list()

            # Format the scores to show them if the f-score is not computed as
            # part of a full evaluation
            if not full_eva:
                f_print = 20*"_" + f"\nF_{beta} scores:\n"
                f1 = ""
                f2 = ""

            # Calculate the f-score for each indivudual label from the label's
            # precision and recall score
            for i, label in enumerate(self.labels):
                try:
                    f = (1 + beta*beta)*self.precision_scores[i]*self.recall_scores[i] / (
                        beta*beta*self.precision_scores[i] + self.recall_scores[i])
                except ZeroDivisionError:
                    # If the label was never predicted correctly, the score is 0
                    f = 0.0
                f_scores.append(f)

                # Only format the results if they are shown immediately as part of this function
                if not full_eva:
                    f1 += label + " "
                    f2 += f"{round(f,4):<{len(label)+1}}"
            if full_eva:
                return f_scores

            # Print all individual f-scores and their average
            f_print += f1 + "Average\n" + f2 + \
                str(round(sum(f_scores)/len(f_scores), 4))
            print(f_print)

            # Add f-scores with the current beta to the evaluation object
            # and return them
            self.f_scores[beta] = f_scores
            return f_scores

    def showConfusionMatrix(self, full_eva: bool = False) -> dict[str,dict[str,int]]:
        """
        Shows the contingencies between the gold and predicted labels as a table.
        Author: Carlotta Quensel

        Args:
            full_eva (bool, optional):
                If the f-score is computed as part of a full evaluation, 
                the results are not shown immediately after computation
                but in the respective function. Defaults to False.

        Returns:
            dict[str,dict[str,int]]: 
                The confusion matrix converted from a list of list into 
                a dictionary to include the row and column labels
        """
        if not self.confusionMatrix:
            print(
                "There are no labels to compute a confusion matrix from, use set_predictions first")
            return None
        # _________________________________STRING FORMATING_________________________________ #
        # Replace author names by A1, A2, ... because 30 names are too long for a clear 
        # and neat table,
        c_labels = len(self.labels)
        # Set the caption with author-abbreviation mapping
        caption = "\n".join(
            [f'A{i:<2} : {self.labels[i]}' for i in range(c_labels)])

        # Set the length of the label columns to the length of all labels
        # combined with a spacer in between each label.
        row_title = len("Gold: ")
        col_len = 4*c_labels

        # Set the column description (predictions) and the column titles
        # as shortened author IDs (A1, A2...) delimited by |
        cm_print = f"Confusion Matrix:\n\n{' ':<{row_title}}{'Predicted:':<{col_len-1}}|\n"
        cm_print += f"{' ':<{row_title}}" + \
            "|".join([f"A{i:<2}" for i in range(c_labels)]) + "|\n"
        # The lines between rows are formed by dashes with + where they cross column lines
        hlines = f"{'-':{'-'}<{row_title-1}}+" + \
            "+".join(["---" for i in range(c_labels)]) + "+\n"
        # Adding the first horizontal line and the axis description for the rows as the gold labels
        cm_print += hlines + "\nGold:|" + \
            "|".join(["   " for i in range(c_labels)]) + "|\n"
        # _________________________________STRING FORMATING_________________________________ #


        # Initialize the Confusion matrix dictionary with the gold labels 
        # as keys (corresponds to rows in the table)
        cm = dict()
        for gold_index in range(c_labels):
            # Initialize the predictions for the current gold label
            gold_label = self.labels[gold_index]
            cm[gold_label] = dict()
            # Add the current author's short ID to the output string
            cm_print += f"A{gold_index:<{row_title-2}}|"
            for predicted_index in range(c_labels):
                # Add the number of documents for which the gold (row) label
                # was predicted as the current (column) label to the dictionary
                # and the output table string
                cm[gold_label][self.labels[predicted_index]
                               ] = self.confusionMatrix[gold_index][predicted_index]
                cm_print += f"{self.confusionMatrix[gold_index][predicted_index]:<3}|"
            # Each row is followed by a horizontal line in the table
            cm_print += "\n" + hlines
        
        cm_print += "\n\nAuthor keys:\n" + caption
        if full_eva:
            # In a full evaluation setting, only return the string of the table to the
            # fullEval method to be printed with all other evaluation scores.
            return cm, cm_print
        # Output the confusion matrix as a readable table and return it as an 
        # interpretable dictionary with the respective labels.
        print(cm_print)
        return cm

    def fullEval(self) -> dict[str,dict[str,int]]:
        """Perform a full evaluation comprised of the Precision, Recall and F1-score
        for each author and a confusion matrix.
        Author: Carlotta Quensel

        Returns:
            dict[str,dict[str,int]]:
                Confusion matrix as a dictionary ordered first by gold label
                and second by predicted label.
        """

        # Compute precision, recall and f1 and their macro average over all labels
        p = self.precision(full_eva=True)
        p.append(sum(p)/len(p))
        r = self.recall(full_eva=True)
        r.append(sum(r)/len(r))
        f1 = self.f_score(full_eva=True)
        f1.append(sum(f1)/len(f1))
        # Get the formatted confusion matrix table
        cm, cm_print = self.showConfusionMatrix(full_eva=True)

        # Calculate the micro average as the accuracy (single-label multi-class classifier):
        # correct classifications / all classifications
        micro = round(sum([self.confusionMatrix[j][j] for j in range(len(self.confusionMatrix))]) /
                      sum([sum(self.confusionMatrix[i]) for i in range(len(self.confusionMatrix))]), 3)

        # Determine the longest author name for output formatting
        title_len = max(map(len, self.labels))+1
        # Add a spacer for the author names in the title of the evaluation
        eva_print = title_len * " " + "Precision " + "Recall " + "F1-Score\n"
        # Add the precision, recall and f-score for each label
        for i in range(len(self.labels)):
            eva_print += f"{self.labels[i]:<{title_len}}{round(p[i],3):<10}{round(r[i],3):<7}{round(f1[i],3)}\n"
        # Add the micro and macro average for all three scores
        eva_print += f"{'Micro-avg':<{title_len}}{micro:<10}{micro:<7}{micro}\n"
        eva_print += f"{'Macro-avg':<{title_len}}{round(p[-1],3):<10}{round(r[-1],3):<7}{round(f1[-1],3)}\n"

        # Add the confusion matrix to the output and show everything
        eva_print += 20*"_" + "\n" + cm_print
        print(eva_print)
        return cm
