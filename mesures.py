# -*- coding: utf-8 -*-
u"""implémente différentes mesures de comparaison."""


def F1_span_overlap(sent1, sent2, label):
    u"""mesure F1 pour 2 phrases encodées BIO MONOLABEL.

    Comparaison par span avec overlap
    Input:
    sent1 prédiction, sent2 réalité, sous forme de listes
    label

    Output:
    truepos, trueneg, falsepos, falseneg
    """
    truepos = 0
    falsepos = 0
    trueneg = 0
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
            bool_trueneg = True
            if cond3 and cond4:
                bool_truepos = True
            elif not cond3 or cond4:
                bool_trueneg = False
        if bool_truepos:
            truepos += 1
        elif bool_trueneg:
            trueneg += 1
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
    return truepos, trueneg, falsepos, falseneg
