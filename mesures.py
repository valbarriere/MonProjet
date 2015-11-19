# -*- coding: utf-8 -*-
u"""implémente différentes mesures de comparaison."""

from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn


""" Ce module contient toutes les fonctions qui permettent de comparer
2 sequences de labels. F1_token compare mot a mot, c'est le truc de base 
a utiliser !

F1_span_overlap c'est une mesure que j'ai developpee, je t'expliquerai le 
principe si tu veux mais c'est pas forcement necessaire que tu t'y interesses.
et BIO_classification_report c'est le built-in NLTK pour comparer
token par token, donc très proche de F1_token, à ceci près qu'il est sensible
a la difference entre B-A et I-A par exemple, alors que dans ma fonction j'ai 
pris soin de l'enlever (pourquoi t'as pris le soin de l'enlever deja ?)."""


def F1_span_overlap(sent1, sent2, label):
    u"""mesure F1 pour 2 phrases encodées BIO MONOLABEL.

    Comparaison par span avec overlap
    Input:
    sent1 prédiction, sent2 réalité-terrain, sous forme de listes
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
    # On procede chunk par chunk de i(t) a j(t) puis de j+1(t) à j(t+1), etc...
    while i < len(sent2):
        while j < len(sent2) and sent2[j][2:] == sent2[i][2:]:
            j += 1
        bool_truepos = False
        for k in list(range(i, j)):
            cond3 = sent1[k][2:] == sent2[i][2:]
            cond4 = sent2[i][2:] == label # pour que ca soit pas 'O'
            if cond3 and cond4:
                bool_truepos = True
        if bool_truepos:
            truepos += 1
        i = j # pour passer au type de tag suivant (car on tag des bouts de phrases)
        
    # second passage pour sent1
    i = 0
    j = 0
    
    while i < len(sent1):
        while j < len(sent1) and sent1[j][2:] == sent1[i][2:]:
            j += 1
        bool_falseneg = False
        bool_falsepos = True
        for k in list(range(i, j)):
            cond3 = sent2[k][2:] == sent1[i][2:]
            cond4 = sent1[i][2:] == label
            if not cond4: # Pour ne s'occuper que d'un certain type de label
                if not cond3: # Prediction a tort
                    bool_falseneg = True
                else: # prediction a raison
                    bool_falsepos = False
        if bool_falseneg:
            falseneg += 1
        elif bool_falsepos:
            falsepos += 1
        i = j
    return truepos, falsepos, falseneg


def F1_token(sent1, sent2, label):
    u"""mesure F1 pour 2 phrases encodées BIO MONOLABEL.

    Comparaison par token DIFFERENT DE O !!
    Input:
    sent1 prédiction, sent2 réalité, sous forme de listes
    label celui qu'on veut (ex attitude)

    Output:
    truepos, falsepos, falseneg
    
    avec label en variable on peut le faire sur tous les labels
    """
    truepos = 0
    falsepos = 0
    falseneg = 0
    # un seul passage, mot par mot
    for i in range(len(sent2)): # sent1[i][2:] on prend pas les 2 premieres (donc BIO)
        if sent1[i][2:] == label and sent1[i][2:] == sent2[i][2:]:
            truepos += 1
        elif sent1[i][2:] == label and sent1[i][2:] != sent2[i][2:]:
            falsepos += 1 # tu as dit qu'il etait bon mais il etait faux
        elif sent1[i][2:] != label and sent1[i][2:] != sent2[i][2:]:
            falseneg += 1 # tu as dit qu'il etait faux mais il etait bon : realite =label qu'on veut
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