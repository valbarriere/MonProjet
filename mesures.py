# -*- coding: utf-8 -*-
u"""implémente différentes mesures de comparaison."""

from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn


def F1_span_overlap(sent1, sent2, label):
    u"""mesure F1 pour 2 phrases encodées BIO MONOLABEL.

    Comparaison par span avec overlap
    Input:
    sent1 prédiction, sent2 réalité, sous forme de listes
    label

    Output:
    truepos, falsepos, falseneg
    """
    truepos = 0
    falsepos = 0
    falseneg = 0
    # premier passage pour sent2
    i = 0
    j = 0
    while i < len(sent2):
        while j < len(sent2) and sent2[j][2:] == sent2[i][2:]:
            j += 1
        for k in list(range(i, j)):
            cond3 = sent1[k][2:] == sent2[i][2:]
            cond4 = sent2[i][2:] == label
            bool_truepos = False
            if cond3 and cond4:
                bool_truepos = True
        if bool_truepos:
            truepos += 1
        i = j
    # second passage pour sent1
    i = 0
    j = 0
    while i < len(sent1):
        while j < len(sent1) and sent1[j][2:] == sent1[i][2:]:
            j += 1
        for k in list(range(i, j)):
            cond3 = sent2[k][2:] == sent1[i][2:]
            cond4 = sent1[i][2:] == label
            bool_falseneg = False
            bool_falsepos = True
            if not cond3 and not cond4:
                bool_falseneg = True
            elif cond3 or not cond4:
                bool_falsepos = False
        if bool_falseneg:
            falseneg += 1
        elif bool_falsepos:
            falsepos += 1
        i = j
    return truepos, falsepos, falseneg

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )