# -*- coding: utf-8 -*-

import nltk

def wordToFeatures(sentence, i):
    word = sentence[i][0]
    postag = sentence[i][1]
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
        word1 = sentence[i-1][0]
        postag1 = sentence[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sentence)-1:
        word1 = sentence[i+1][0]
        postag1 = sentence[i+1][1]
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

def sentenceToFeatures(sentence):
    return [wordToFeatures(sentence, i) for i in range(len(sentence))]

def toFeatures(f2_file):
    """
        retourne les features pour le document selectionné
    """
    
    features_file = []
    for f2_sentence in f2_file:
        list_sentence = []
        for f2_word in f2_sentence:
            listlab_word = f2_word.split(";")
            list_sentence.append(listlab_word[0])
        pos_sentence = nltk.pos_tag(list_sentence)
        features_file.append(sentenceToFeatures(pos_sentence))
        
    return features_file
    
def toLabels(f2_file,int_list):
    """
        retourne les labels pour le document selectionné
    """
    
    labels_file = []
    for i in int_list:
        labelI_file = []         
        for f2_sentence in f2_file:
            list_sentence = []
            for f2_word in f2_sentence:
                listlab_word = f2_word.split(";")
                list_sentence.append(listlab_word[i])
            labelI_file.append(list_sentence)
        labels_file.append(labelI_file)
            
    return labels_file    