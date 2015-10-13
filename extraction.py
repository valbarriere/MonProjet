# -*- coding: utf-8 -*-
u"""méthodes pour extraire les features/labels à partir du dump."""

from nltk.corpus import sentiwordnet as swn

import re
import os
import sys
import nltk
import numpy as np

MULTILABEL = ('B-evaluation', 'B-affect', 'I-evaluation', 'I-affect',
              'B-source', 'I-source', 'B-target', 'I-target')
MORPHY_TAG = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
HIERARCHY = {'I-attitude_positive': 1, 'B-attitude_positive': 2, 'I-attitude_negative': 3, 'B-attitude_negative': 4, 'I-source': 5, 'B-source': 6,
             'I-target': 7, 'B-target': 8, 'O': 9}
D_PATH = '/home/lucasclaude3/Documents/Stage_Telecom/'


""" Peut être le module en apparence le plus bordelique à cause de sa structure
hierarchique. Il permet d'extraire les features et les labels a partir des dump.
Pour bien comprendre ce qui se passe il faut partir de "exctrat2CRFsuite" et
remonter à chaque fois qu'il y a des appels de fonction.

Tu retrouveras aussi "count_labels" et "stats_labels" qui permettent de compter
le nombre d'occurences des labels et de produire les stats dessus."""


def read_patterns(path):
    u"""Lit les patterns détectés par les règles syntaxiques de Caro.
    
    Retourne un dict avec les patterns comme cles.
    """
    dict_patterns = {}
    f = open(path, 'Ur')
    text = f.read()
    sents = text.split('\n')
    for sent in sents:
        elements = sent.split(';')
        try:
            if elements[0] == 'session':
                continue
            
            sent = " ".join(nltk.word_tokenize(elements[2]))
            if not dict_patterns.__contains__(sent):
                dict_patterns[sent] = nltk.word_tokenize(elements[4])
        except IndexError:
            print("End of patterns file")        
    return dict_patterns

PATTERNS = read_patterns('patterns.csv')


def __merge_dicts(*dict_args):
    u"""Fusionne n'importe quel nombre de dict."""
    z = {}
    for y in dict_args:
        z.update(y)
    return z


def __word2features(sent, i):
    u"""Features lexicaux et syntaxiques."""
    word = sent[i][0].lower()
    postag = sent[i][1][:2]
    features = {
        'bias': 1.0,
        'word': word,
        'postag': postag
    }
    if i > 0:
        word1 = sent[i-1][0].lower()
        postag1 = sent[i-1][1][:2]
        features.update({
            '-1:word': word1,
            '-1:postag': postag1
        })
    else:
        features['BOS']=1.0


    if i > 1:
        word1 = sent[i-2][0].lower()
        postag1 = sent[i-2][1][:2]
        features.update({
            '-2:word': word1,
            '-2:postag': postag1
        })
    else:
        features['B2OS']=1.0


    if i < len(sent)-1:
        word1 = sent[i+1][0].lower()
        postag1 = sent[i+1][1][:2]
        features.update({
            '+1:word': word1,
            '+1:postag': postag1
        })
    else:
        features['EOS']=1.0

    if i < len(sent)-2:
        word1 = sent[i+2][0].lower()
        postag1 = sent[i+2][1][:2]
        features.update({
            '+2:word': word1,
            '+2:postag': postag1
        })
    else:
        features['E2OS']=1.0


    boolVP = False
    for j in range(len(sent)):
        if sent[j][1][:2] == 'VB':
            boolVP = True
            features['phrase_type']='VP'
    if boolVP is False:
        features['phrase_type']='NP'

    try:
        tag_conversion = MORPHY_TAG[postag]
        synset = list(swn.senti_synsets(word, pos=tag_conversion))[0]
        polarity = [synset.pos_score(), synset.neg_score(), synset.obj_score()]
        features.update({
            'synset.pos_score()': polarity[0],
            'synset.neg_score()': polarity[1],
            'synset.obj_score()': polarity[2]
        })
    except (KeyError, IndexError):
        pass
    
    return features


def __audio2features(audio, i):
    dict_pitch = eval(audio[i])
    result_pitch = {}
    for k, v in dict_pitch.items():
        if dict_pitch[k] != None:
            result_pitch[k] = v
    return result_pitch


def __rules2features(sent, i):
    result = {}
    formated_sent = " ".join([sent[k][0] for k in range(len(sent))])
    if formated_sent in PATTERNS:
        result['inRule'] = 1.0
        target = PATTERNS[formated_sent]
        if sent[i][0] in target:
            result['inTarget'] = 1.0
    return result
    

def __sent2features(sent, audio, mfcc):
    u"""Choisir les types de features utilisés ici.
    
    Il n'y a qu'a fusionner les dict voulus.
    """
    return [__merge_dicts(__word2features(sent, i),
                          __audio2features(audio,i)) for i in range(len(sent))]


def __sent2label(sent, label):
    return [__decision(str_labels, label) for token, postag,
            str_labels in sent]


def __sent2tokens(sent):
    return [token for token, postag, label in sent]


def __decision(str_labels, label):
    list_labels = str_labels.split(";")
    if label == 'BIO':
        list_nb = [HIERARCHY[lab] for lab in list_labels]
        return "".join([k for k, v in HIERARCHY.items()
                        if v == np.min(list_nb)])
    else:
        if "I-"+label in list_labels:
            return "I-"+label
        elif "B-"+label in list_labels:
            return "B-"+label
        else:
            return "O"


def text_sents(path):
    u"""Traite le texte."""
    f = open(path, 'Ur')
    sents = f.read().split('\n\n\n')
    sents_list = []
    for sent in sents:
        words = sent.split('\n')
        words_list = []
        for word in words:
            features = tuple(word.split('\t'))[:2]
            words_list.append(features)
        sents_list.append(words_list)
    return sents_list
    

def audio_sents(path):
    u"""Traite l'audio."""
    f = open(path, 'Ur')
    sents = f.read().split('\n\n\n')
    sents_list = []
    for sent in sents:
        words = sent.split('\n')
        words_list = []
        for word in words:
            try:
                features = word.split('\t')[1]
                if features == 'None':
                    features = "{}"
                words_list.append(features)
                m = re.findall(r"\>|\<|'|GONNA|WANNA", word.split('\t')[0])
                for k in range(len(m)):
                    words_list.append(features)
            except IndexError:
                print('END OF FILE %s' % path[-3:])
                break
        sents_list.append(words_list)
    return sents_list  


def extract2CRFsuite(path_text, path_audio, path_mfcc, label='BIO'):
    u"""PLUS IMPORTANTE.
    
    Extrait features et label pour une session
    à partir d'un dossier contenant les dump au format Conll
    """
    text = nltk.corpus.conll2002.iob_sents(path_text)
    audio = audio_sents(path_audio)
    mfcc = audio_sents(path_mfcc)       
    X = [__sent2features(s, t, u) for (s, t, u) in zip(text, audio, mfcc)]
    y = [__sent2label(s, label) for s in text]
    return X, y


def count_labels(path, dump_filename):
    u"""Compte le nombre d'occurrences des combinaisons de labels.

    path renvoie au dossier contenant les dump.
    dump_filename est le nom du fichier où seront stockés les stats.
    """
    dict_multilabels = {}
    dict_cpt = {}
    for filename in os.listdir(path):
        if os.path.isdir(path+filename):
            continue
        sents = nltk.corpus.conll2002.iob_sents(path+filename)
        for sent in sents:
            for token, postag, str_labels in sent:
                set_labels = set(str_labels.split(";"))
                if set_labels not in dict_multilabels.values():
                    i = len(dict_multilabels)
                    dict_multilabels[i] = set_labels
                    dict_cpt[i] = 1
                else:
                    i = "".join([str(k) for k, v in dict_multilabels.items()
                                 if v == set_labels])
                    dict_cpt[int(i)] += 1
    f = open(dump_filename, 'w')
    for j in range(len(dict_multilabels)):
        f.write("%s\t%d\n" % (dict_multilabels[j], dict_cpt[j]))
    f.close()


def labels_stats(dump_filename, stats_filename):
    u"""Classe les labels par fréquence."""
    f = open(dump_filename, 'r')
    occurrences = []
    total_wO = 0
    total_woO = 0
    dict_nb = {}
    while 1:
        line = f.readline()
        if line == "":
            break
        multi_lab, cpt = line.split("\t")
        occurrences.append((eval(multi_lab), int(cpt)))
        total_wO += int(cpt)
        if eval(multi_lab) == {'O'}:
            dict_nb[0] = int(cpt)
        else:
            total_woO += int(cpt)
            if len(eval(multi_lab)) not in dict_nb:
                dict_nb[len(eval(multi_lab))] = int(cpt)
            else:
                dict_nb[len(eval(multi_lab))] += int(cpt)
    f.close()
    ranking = sorted(occurrences, key=lambda data: data[1], reverse=True)
    f = open(stats_filename, 'w')
    f.write("labels\toccurrences\tfrequencies\tfrequencies without O\n")
    for i in range(len(ranking)):
        if i == 0:
            f.write("%s\t%d\t%f\n" % (ranking[i][0],
                                      ranking[i][1],
                                      ranking[i][1]/total_wO * 100))
        else:
            f.write("%s\t%d\t%f\t%f\n" % (ranking[i][0],
                                          ranking[i][1],
                                          ranking[i][1]/total_wO * 100,
                                          ranking[i][1]/total_woO * 100))
    f.write("\nnb_labels\toccurrences\tfrequencies\tfrequencies without O\n")
    for key in sorted(dict_nb):
        if key == 0:
            f.write("%s\t%d\t%f\n" % (key,
                                      dict_nb[key],
                                      dict_nb[key]/total_wO * 100))
        else:
            f.write("%s\t%d\t%f\t%f\n" % (key,
                                          dict_nb[key],
                                          dict_nb[key]/total_wO * 100,
                                          dict_nb[key]/total_woO * 100))
    f.close()
    
if __name__ == "__main__":
    count_labels(D_PATH+'Datasets/Semaine/all/dump_attitudeposneg_only/',
                 D_PATH+'MonProjet/stats/labels_occurrences')
    labels_stats(D_PATH+'MonProjet/stats/labels_occurrences',
                 D_PATH+'MonProjet/stats/labels_stats')
