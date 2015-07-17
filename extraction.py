# -*- coding: utf-8 -*-
u"""méthodes pour extraire les features/labels à partir du dump."""

from nltk.corpus import sentiwordnet as swn

import os
import nltk

MULTILABEL = ('B-evaluation', 'B-affect', 'I-evaluation', 'I-affect',
              'B-source', 'I-source', 'B-target', 'I-target')
MORPHY_TAG = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}


def __word2features(sent, i):
    u"""rajouter des features globaux : synsets, phrase verbale ou nominale."""
    word = sent[i][0].lower()
    postag = sent[i][1][:2]
    features = [
        'bias',
        'word=' + word,
        'postag=' + postag,
    ]
    if i > 0:
        word1 = sent[i-1][0].lower()
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word=' + word1,
            '-1:postag=' + postag1,
        ])
    else:
        features.append('BOS')

    if i > 1:
        word1 = sent[i-2][0].lower()
        postag1 = sent[i-2][1]
        features.extend([
            '-2:word=' + word1,
            '-2:postag=' + postag1,
        ])
    else:
        features.append('B2OS')

    if i < len(sent)-1:
        word1 = sent[i+1][0].lower()
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1,
            '+1:postag=' + postag1,
        ])
    else:
        features.append('EOS')

    if i < len(sent)-2:
        word1 = sent[i+2][0].lower()
        postag1 = sent[i+2][1]
        features.extend([
            '+2:word=' + word1,
            '+2:postag=' + postag1,
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
        tag_conversion = MORPHY_TAG[postag]
        synset = swn.senti_synsets(word, pos=tag_conversion)
        polarity = [synset.pos_score(), synset.neg_score(), synset.obj_score()]
        features.extend([
            'synset.pos_score()=' + str(int(polarity[0]*2)),
            'synset.neg_score()=' + str(int(polarity[1]*2)),
            'synset.obj_score()=' + str(int(polarity[2]*2)),
            'polarity=' + str(polarity.index(max(polarity)))
        ])
    except:
        pass

    return features


def __sent2features(sent):
    return [__word2features(sent, i) for i in range(len(sent))]


def __sent2label(sent, label):
    return [__decision(str_labels, label) for token, postag,
            str_labels in sent]


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


def extract2CRFsuite(path, label):
    u"""Extrait un dataset au format utilisable par CRFsuite.

    à partir d'un dossier contenant les dump au format Conll
    """
    X = []
    y = []
    for filename in os.listdir(path):
        train_sents = nltk.corpus.conll2002.iob_sents(path+"/"+filename)
        X = X + [__sent2features(s) for s in train_sents]
        y = y + [__sent2label(s, label) for s in train_sents]
    return X, y
