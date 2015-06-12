# -*- coding: utf-8 -*-

import os
import nltk

def word2features(sent, i):
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
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2label(sent,label):
    return [decision(str_labels,label) for token, postag, str_labels in sent]
    
def sent2labels(sent,label):
    return [multilabel(str_labels) for token, postag, str_labels in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]
    
def decision(str_labels,label):
    list_labels = str_labels.split(";")
    if "I"+label in list_labels:
        return "I"+label
    elif "B"+label in list_labels:
        return "B"+label
    else:
        return "N"
        
def multilabel(str_labels):
    return "N"
    
def extract2CRFsuite(path):
    """
    Extrait un dataset au format utilisable par CRFsuite 
    à partir d'un dossier contenant les dump au format Conll
    """
    
    X = []
    y = []
    for filename in os.listdir(path):
        train_sents = nltk.corpus.conll2002.iob_sents(path+"/"+filename)
        X.append([sent2features(s) for s in train_sents])
        y.append([sent2label(s,"evaluation") for s in train_sents])
    
    return X, y
        