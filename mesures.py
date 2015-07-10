# -*- coding: utf-8 -*-
u"""implémente différentes mesures de comparaison."""


def exact_comparison(sent1, sent2):
    u"""mesure exacte pour 2 phrases.

    Input:
    sent1, sent2 sous forme de listes
    """
    if len(sent1) != len(sent2):
        raise Exception("Les 2 phrases ne correspondent pas.")
    else:
        exact = 1
        for i in range(len(sent1)):
            if sent1[i] != sent2[i]:
                exact = 0
    return exact


def overlap_comparison(sent1, sent2):
    u"""mesure overlap pour 2 phrases encodées BIO MULTILABEL.

    Input:
    sent1, sent2 sous forme de listes
    """
    if len(sent1) != len(sent2):
        raise Exception("Les 2 phrases ne correspondent pas.")
    else:
        overlap = 0
        for i in range(len(sent1)):
            if set(sent1[i]) == set(sent2[i]):
                overlap += 1
        return overlap/len(sent1)


def F1_comparison(sent1, sent2, label):
    u"""mesure F1 pour 2 phrases encodées BIO MONOLABEL.

    Input:
    sent1, sent2 sous forme de listes
    label

    Output:
    truepos, trueneg, falsepos, falseneg
    """
    if len(sent1) != len(sent2):
        raise Exception("Les 2 phrases ne correspondent pas.")
    else:
        truepos = 0
        falsepos = 0
        trueneg = 0
        falseneg = 0
        for i in range(len(sent1)):
            lab1 = sent1[i][2:]
            lab2 = sent2[i][2:]
            cond1 = lab1 == lab2
            cond2 = lab1 == label
            if cond1 and cond2:
                truepos += 1
            if cond1 and not cond2:
                trueneg += 1
            if not cond1 and cond2:
                falsepos += 1
            else:
                falseneg += 1
    return truepos, trueneg, falsepos, falseneg


def global_comparison(dump1, dump2, method='exact'):
    u"""détermine le score de détection global.

    Input:
    dump1, dump2: chemins des fichiers txt contenant les taggings par phrase.
    method: chaine de caractère indiquant la méthode désirée.
    """
    pass
