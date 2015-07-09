u"""méthodes pour extraire les features/labels à partir du dump."""
# -*- coding: utf-8 -*-

from nltk.corpus import wordnet as wn

import os
import nltk

MULTILABEL = ('B-evaluation', 'B-affect', 'I-evaluation', 'I-affect',
              'B-source', 'I-source', 'B-target', 'I-target')
MORPHY_TAG = {'NN': wn.NOUN, 'JJ': wn.ADJ, 'VB': wn.VERB, 'RB': wn.ADV}


def word2features(sent, i):
    u"""rajouter des features globaux : synsets, phrase verbale ou nominale."""
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i > 1:
        word1 = sent[i-2][0]
        postag1 = sent[i-2][1]
        features.extend([
            '-2:word.lower=' + word1.lower(),
            '-2:word.istitle=%s' % word1.istitle(),
            '-2:word.isupper=%s' % word1.isupper(),
            '-2:postag=' + postag1,
            '-2:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('B2OS')

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    if i < len(sent)-2:
        word1 = sent[i+2][0]
        postag1 = sent[i+2][1]
        features.extend([
            '+2:word.lower=' + word1.lower(),
            '+2:word.istitle=%s' % word1.istitle(),
            '+2:word.isupper=%s' % word1.isupper(),
            '+2:postag=' + postag1,
            '+2:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('E2OS')

    boolVP = False
    for j in range(len(sent)):
        if sent[j][1][:2] == 'VB':
            boolVP = True
            features.append('phrase_type=VP')
    if boolVP is False:
        features.append('phrase_type=NP')

    try:
        v = MORPHY_TAG[postag]
        w = wn.synsets(word, pos=v)
    except KeyError:
        w = wn.synsets(word)
    if len(w) > 0:
        features.append('hypernym=' + str(w[0].root_hypernyms()))

    return features


def __sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def __sent2label(sent, label):
    return [__decision(str_labels, label) for token, postag,
            str_labels in sent]


def __sent2labels(sent):
    return [__multilabel(str_labels) for token, postag, str_labels in sent]


def __sent2tokens(sent):
    return [token for token, postag, label in sent]


def __decision(str_labels, label):
    list_labels = str_labels.split(";")
    if "I-"+label in list_labels:
        return "I-"+label
    elif "B-"+label in list_labels:
        return "B-"+label
    else:
        return "O"


def __multilabel(str_labels):
    str_binary = ""
    list_labels = str_labels.split(";")
    for i in range(len(MULTILABEL)):
        if MULTILABEL[i] in list_labels:
            str_binary += '1'
        else:
            str_binary += '0'
    return str_binary


def extract2CRFsuite(path, label="all"):
    u"""Extrait un dataset au format utilisable par CRFsuite.

    à partir d'un dossier contenant les dump au format Conll
    """
    X = []
    y = []
    for filename in os.listdir(path):
        train_sents = nltk.corpus.conll2002.iob_sents(path+"/"+filename)
        X = X + [__sent2features(s) for s in train_sents]
        if label == "all":
            y = y + [__sent2labels(s) for s in train_sents]
        else:
            y = y + [__sent2label(s, label) for s in train_sents]
    return X, y
