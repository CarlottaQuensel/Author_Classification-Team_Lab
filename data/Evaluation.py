# -*- coding: utf-8 -*-

class Evaluation():
    """Evaluation for a trained multi-class single-label classifier.

    Methods:
    recall - compute the recall for every or a specific label
    precision - compute the precision for every or a specific label
    f1_score - compute the balanced f-score (alpha=0.5) from recall and precision
    confusion_matrix - erroneous classification as contingency table between gold and predicted labels
    """
    precision_scores = dict()
    recall_scores = dict()
    f_scores = dict()

    def __init__(self, gold_labels: list[str], predicted_labels: list[str]):
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
            predicted_labels (list(str)): 
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
            true = self.labels.index(self.gold[i])
            pred = self.labels.index(self.predicted[i])
            confusion[true][pred] += 1
        return confusion


    def precision(self, label: str = None, full_eva: bool = False) -> list[float]:
        """Compute precision for one or all labels as the ratio of predicted labels that are true

        Args:
            label (string, optional): 
                If set, only the precision score for this label is computed and shown. Defaults to None.
            full_eva (boolean, optional): 
                If the precision is computed as part of a full evaluation, the results are shown in the respective function and not immediately

        Returns:
            list[float]: The precision scores for each label in the same order as the class' label set
        """
        # If the user wants to see the precision for a specific author, only compute this value
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
                print(f"Precision of {label}: {round(prec,4)}")
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
                try:
                    prec_scores.append(tp/(tp+fp))
                except ZeroDivisionError:
                    prec_scores.append(1)
            self.prec = prec_scores

            # When a full evaulation is performed, the results are not shown directly after computing them but rather
            # passed on to the full evaluation function to be shown together with the other scores. 
            if full_eva:
                self.precision_scores = prec_scores
                return prec_scores
            # Show the precision for each label and their average (micro-F1)
            prec_print = 20*"_" + "\nPrecision:\n"
            p1 = ""
            p2 = ""
            for i in range(len(prec_scores)):
                p1 += f"{self.labels[i]} "
                filler = len(self.labels[i])+1
                p2 += f"{round(prec_scores[i], 4) : <{filler}}"
            av = sum(prec_scores)/len(prec_scores)
            prec_print += p1 + "Average\n" + p2 + str(round(av,4))
            print(prec_print)
            self.precision_scores = prec_scores
            return prec_scores

    
    def recall(self, label=None, full_eva: bool = False) -> list[float]:
        """Compute the recall for one or all labels as the ratio of gold labels that are predicted and show the scores.

        Args:
            label (string, optional): 
                If set, only the recall score for this label is computed and shown. Defaults to None.
            full_eva (bool, optional): 
                If the recall is computed as part of a full evaluation, the scores are shown with all others and not immediately.
                Defaults to False.

        Returns:
            list[float]: Recall scores for each label in the same order as the class' label set
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
                try:
                    rec_scores.append( tp / (tp+fn) )
                except ZeroDivisionError:
                    rec_scores.append( 0 )
            self.rec = rec_scores

            # If a full evaluation is done, the results are not shown immediately but rather passed up to the respective function to be
            # shown together with all other scores
            if full_eva:
                self.recall_scores = rec_scores
                return rec_scores
            
            # Show the recall for each label, their average
            rec_print = 20*"_" + "\nRecall:\n"
            r1 = ""
            r2 = ""
            for i in range(len(rec_scores)):
                r1 += f"{self.labels[i]} "
                filler = len(self.labels[i])+1
                r2 += f"{round(rec_scores[i],4):<{filler}}"
            av = sum(rec_scores)/len(rec_scores)
            rec_print += r1 + "Average\n" + r2 + str(round(av,4))
            print(rec_print)
            self.recall_scores = rec_scores
            return rec_scores
    

    def f_score(self, label: str = None, beta: int = 1, full_eva: bool = False) -> list[float]:
        """Compute and show the f score as a combination of precision and recall. Per default, their harmonic mean is computed.

        Args:
            label (str, optional): 
                Only compute and show the score for a specific label. Defaults to None.
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
                if full_eva:
                    return self.f_scores[beta]
                f_print = 20*"_" + f"\nF_{beta} scores\n"
                f1 = ""
                f2 = ""
                for i, label in enumerate(self.labels):
                    f1 += label + " "
                    f = self.f_scores[beta][i]
                    f2 += f"{round(f,4):<{len(label)+1}}"
                f_print += f1 + "Average\n" + f2 + str(round(sum(self.f_scores[beta])/len(self.f_scores[beta]),4))
                print(f_print)
                return self.f_scores[beta]

            # Calculate precision and recall if not already done
            if not len(self.precision_scores) or not len(self.recall_scores):
                self.precision()
                self.recall()
            
            # List of F-scores for all labels
            f_scores = list()
            # Strings for showing the output immediately after computation 
            if not full_eva:
                f_print = 20*"_" + f"\nF_{beta} scores:\n"
                f1 = ""
                f2 = ""
            # Calculate the f-score for each indivudual label
            for i, label in enumerate(self.labels):
                try:
                    f = (1+ beta*beta)*self.precision_scores[i]*self.recall_scores[i] / (beta*beta*self.precision_scores[i] + self.recall_scores[i])
                except ZeroDivisionError:
                    f = 0
                f_scores.append(f)
                # Ignore formating if the result is not shown immediately as part of this function
                if not full_eva:
                    f1 += label + " "
                    f2 += f"{round(f,4):<{len(label)+1}}"
            if full_eva:
                return f_scores
            # Print all individual f-scores and their average
            f_print += f1 + "Average\n" + f2 + str(round(sum(f_scores)/len(f_scores),4))
            print(f_print)
            # Add the f-scores with the current beta to the evaluation object and return them
            self.f_scores[beta] = f_scores
            return f_scores

    def showConfusionMatrix(self, full_eva: bool = False) -> dict[dict[int]]:
        if not self.confusionMatrix:
            print("There are no labels to compute a confusion matrix from, use set_predictions first")
            return None
        # _________________________________STRING FORMATING_________________________________ #
        # As the 20-40 author names are too long to use for a clear and neat table, they are replaced by A1,...An
        # with a table caption showing the mapping
        c_labels = len(self.labels)
        caption = "\n".join([f'A{i:<2} : {self.labels[i]}' for i in range(c_labels)])

        # The string for the label columns is as long as all labels combined with a spacer in between each label.
        row_title = len("Gold: ")
        col_len = 4*c_labels


        # The column title of the table consists of the axis description as predicted labels and the shortened author IDs
        # (A1, A2...), which are delimited by | as a vertical line
        cm_print = f"Confusion Matrix:\n\n{' ':<{row_title}}{'Predicted:':<{col_len-1}}|\n"
        cm_print += f"{' ':<{row_title}}" + "|".join([f"A{i:<2}" for i in range(c_labels)]) + "|\n"
        # The lines between rows are formed by dashes with + where they cross column lines
        hlines = f"{'-':{'-'}<{row_title-1}}+" + "+".join(["---" for i in range(c_labels)]) + "+\n"
        # Adding the first horizontal line and the axis description for the rows as the gold labels
        cm_print += hlines + "\nGold:|" + "|".join(["   " for i in range(c_labels)]) + "|\n"
        # _________________________________STRING FORMATING_________________________________ #

        # Each gold label has its own entry in a dictionary and its own row in the printed table
        cm = dict()
        for gold_index in range(c_labels):
            # Initializing the predictions for the current gold label
            gold_label = self.labels[gold_index]
            cm[gold_label] = dict()
            # Adding the author ID for the current author label to the output string
            cm_print += f"A{gold_index:<{row_title-2}}|"
            for predicted_index in range(c_labels):
                # Adding the number of documents for which the gold (row) label was predicted as the current (column) label
                # to the output dictionary and string
                cm[gold_label][self.labels[predicted_index]] = self.confusionMatrix[gold_index][predicted_index]
                cm_print += f"{self.confusionMatrix[gold_index][predicted_index]:<3}|"
            # Each row is followed by a horizontal line in the table
            cm_print += "\n" + hlines
        '''for i, row_label in enumerate(self.labels):
            # Initializing the predictions for the current gold label
            cm[row_label] = dict()
            # Adding the row title to the output string
            row_string = f"{row_label:<{max_label}}|"
            for j, col_label in enumerate(self.labels):
                # Adding the number of times, the gold row label was predicted as the current column label
                cm[row_label][col_label] = self.confusionMatrix[i][j]
                # Adding the number as a cell of the printed table
                row_string += f"{self.confusionMatrix[i][j]:<{len(col_label)}}|"
            # Adding the whole formated row and a horizontal line to the printed table
            cm_print += row_string + "\n" + hlines + "\n"'''
        
        # After the whole matrix is traversed, it is printed in a readable format and returned as an interpretable dictionary with the respective labels
        # If the confusion matrix is shown as part of a full evaluation, the string of the table is returned to the fullEval function to be printed 
        # with all other evaluation scores.
        cm_print += "\n\nAuthor keys:\n" + caption
        if full_eva:
            return cm, cm_print
        print(cm_print)
        return cm


    def fullEval(self) -> dict[dict[int]]:
        """Perform a full evaluation comprised of the Precision, Recall and F1-score for each author and a confusion matrix.

        Returns:
            dict[dict[int]]: 
                Confusion matrix as a dictionary ordered first by gold label and second by predicted label.
        """

        # Compute precision, recall and f1 and their macro average over all labels
        p = self.precision(full_eva = True)
        p.append(sum(p)/len(p))
        r = self.recall(full_eva = True)
        r.append(sum(r)/len(r))
        f1 = self.f_score(full_eva = True)
        f1.append(sum(f1)/len(f1))
        # Get the nicely formated confusion matrix table
        cm, cm_print = self.showConfusionMatrix(full_eva=True)

        # The micro average of the precision, recall and (due to beta=1/harmonic mean) f-scores corresponds to the accuracy
        # for this single-label multi-class classifier, and is computed as (correct classifications / all classifications):
        micro = round(sum([self.confusionMatrix[j][j] for j in range(len(self.confusionMatrix))]) / sum([sum(self.confusionMatrix[i]) for i in range(len(self.confusionMatrix))]),3)

        # To print the scores of all labels and micro/macro average, the longest author name has to be determined
        title_len = max( map(len, self.labels) )+1
        eva_print = title_len * " " + "Precision " + "Recall " + "F1-Score\n"
        # Show the precision, recall and f-score for each label
        for i in range(len(self.labels)):
            eva_print += f"{self.labels[i]:<{title_len}}{round(p[i],3):<10}{round(r[i],3):<7}{round(f1[i],3)}\n"
        # Show the micro and macro average for all three scores
        eva_print += f"{'Micro-avg':<{title_len}}{micro:<10}{micro:<7}{micro}\n"
        eva_print += f"{'Macro-avg':<{title_len}}{round(p[-1],3):<10}{round(r[-1],3):<7}{round(f1[-1],3)}\n"

        # Add the confusion matrix to the output and show everything
        eva_print += 20*"_" + "\n" + cm_print
        print(eva_print)
        return cm
