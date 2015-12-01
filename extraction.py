# -*- coding: utf-8 -*-
u"""méthodes pour extraire les features/labels à partir du dump."""

from features_text import __word2features

import re
import os
#import sys
import nltk
import numpy as np

MULTILABEL = ('B-evaluation', 'B-affect', 'I-evaluation', 'I-affect',
              'B-source', 'I-source', 'B-target', 'I-target')
              
HIERARCHY = {'I-attitude_positive': 1, 'B-attitude_positive': 2, 'I-attitude_negative': 3, 'B-attitude_negative': 4, 'I-source': 5, 'B-source': 6,
             'I-target': 7, 'B-target': 8, 'O': 9}
# D_PATH = '/home/lucasclaude3/Documents/Stage_Telecom/'
D_PATH = "/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/"

""" Peut être le module en apparence le plus bordelique à cause de sa structure
hierarchique. Il permet d'extraire les features et les labels a partir des dump.
Pour bien comprendre ce qui se passe il faut partir de "exctrat2CRFsuite" et
remonter à chaque fois qu'il y a des appels de fonction.

Tu retrouveras aussi "count_labels" et "stats_labels" qui permettent de compter
le nombre d'occurences des labels et de produire les stats dessus."""



def __merge_dicts(*dict_args):
    u"""Fusionne n'importe quel nombre de dict."""
    z = {}
    for y in dict_args:
        z.update(y)
    return z


def __audio2features(audio, i):
    """
    STRING --> DICTIONNAIRE pour un seul mot (le i)
    Permet de mettre le dictionnaire qui n'etait qu'une string sous vrai forme de dictionnaire
    i est le numero du mot, on obtient donc toutes les features audio pour un seul mot
    """

    dict_pitch = eval(audio[i])
    # result au lieu de dict : refaire dico en enlevant les None
    result_pitch = {}
    for k, v in dict_pitch.items():
        if dict_pitch[k] != None:
            result_pitch[k] = v
    return result_pitch


def __sent2features(sent, audio, mfcc, params):
    u"""Choisir les types de features utilisés ici.
    Avec opt, on ne garde qu'audio ou texte si on veut faire des test séparement.
    Il n'y a qu'a fusionner les dict voulus pour chaque mot
    
    sent, audio et mfcc sont d'une seule phrase --> len(sent) est le nbr de mot i le numero du mot
    """
    opt = params['opt']
    if opt == 'MULTI':
        return [__merge_dicts(__word2features(sent, i, params),
                          __audio2features(audio,i)) for i in range(len(sent))] 
    elif opt == 'AUDIO':             
        return [__merge_dicts(__audio2features(audio,i)) for i in range(len(sent))] 
    else: # then it's just text
        return [__merge_dicts(__word2features(sent, i, params)) for i in range(len(sent))] 

def __sent2label(sent, label):
    """
    Return a list with the labels of eahc word in 1 sentence
    """
    return [__decision(str_labels, label) for token, postag,
            str_labels in sent]


def __sent2tokens(sent):
    """List of the words from a sentence
    NOT USED    
    """
    return [token for token, postag, label in sent]


def __decision(str_labels, label):
    """
    label peut etre attitude, source ou target : pour un entrainement séparé 
    mais qui est moins efficace que s'il est bien fait ensemble
    """
    list_labels = str_labels.split(";") # S'il y a plusieurs labels par mot
    if label == 'BIO': # garder toutes les annotations 
        
#        rappel : HIERARCHY = {'I-attitude_positive': 1, 'B-attitude_positive': 2, 'I-attitude_negative': 3, 'B-attitude_negative': 4, 'I-source': 5, 'B-source': 6,
#             'I-target': 7, 'B-target': 8, 'O': 9}
             
        list_nb = [HIERARCHY[lab] for lab in list_labels] # donne un "rang" aux differents labels du mot
        return "".join([k for k, v in HIERARCHY.items() # regarde les rangs des differents labels
                        if v == np.min(list_nb)]) # label qui a le "rang" le plus eleve (nb le + petit) gagne
 
    else: # Si pas BIO, c'est attitude par exemple, on les entraine separement !! (d'abord attitude ou source ou ?)
        
        if label.__class__ == list: # plusieurs labels de sortie --> ex : ['attitde_positive' ,'attitude_negative']
            for k in range(len(label)):            
                if "I-"+label[k] in list_labels:
                    return "I-"+label[k]
                elif "B-"+label[k] in list_labels:
                    return "B-"+label[k]    
                    
            return "O"# if no label
        else: # only one label
            if "I-"+label in list_labels:
                return "I-"+label
            elif "B-"+label in list_labels:
                return "B-"+label
            else:
                return "O"

def text_sents(path):
    u"""Traite le texte. 
    Pas utilisé, on le fait avec nltk.corpus.conll2002.iob_sents(path_text)
    """
    f = open(path, 'Ur')
    sents = f.read().split('\n\n\n') # Phrase 
    sents_list = []
    for sent in sents:
        words = sent.split('\n') # Mots 
        words_list = []
        for word in words:
            features = tuple(word.split('\t'))[:2]
            words_list.append(features)
        sents_list.append(words_list)
    return sents_list
    

def audio_sents(path):
    u"""Traite l'audio.
    Va cherche les dumps et les met dans des variables
    /Datasets/Semaine/all/+ dump_audio/ ou dump_mfcc/    
    """
    f = open(path, 'Ur')
    sents = f.read().split('\n\n\n') # sent[0] 1ere phrase
    sents_list = []
    for sent in sents: # pour chaque phrase
        words = sent.split('\n') # separation par mot words[0] 1er mot
        words_list = []
        for word in words:
            try:
                features = word.split('\t')[1] # Separe le mot ex 'HI' des features (u'moy_loc_B1': -0.059, u'moy_loc_B2': -0.199)
                if features == 'None':
                    features = "{}"
                words_list.append(features)
                m = re.findall(r"\>|\<|'|GONNA|WANNA", word.split('\t')[0]) # trouve les gonna/wanna/'/ qui font 2 mots avc POSTAG 
                #  --> don't = do not, pour l'audio on met les memes features audio pour les 2 mots
                for k in range(len(m)):
                    words_list.append(features)
            except IndexError:
              #  print('END OF FILE %s' % path.split('.')[0][-3:]) #fin du fichier, donne le nom de la session en +
                break
        sents_list.append(words_list)
    return sents_list  


def extract2CRFsuite(path_text, path_audio, path_mfcc, label='BIO', params = None):
    u"""PLUS IMPORTANTE.
    
    Extrait features et label pour une session
    à partir d'un dossier contenant les dump au format Conll
    """
    # Just to charge the good ones : 
    text = None
    audio = None
    mfcc = None
    opt = params['opt']
    text = nltk.corpus.conll2002.iob_sents(path_text) # text[phrase][mot] = (mot, genre NN, BIO-attitude)
    
    # Labels first sice we need the text   
    y = [__sent2label(s, label) for s in text]    
    
    # Then the variables
    if opt == 'TEXT':
        audio = [None]*len(text)
        mfcc =  [None]*len(text)
    elif opt == 'AUDIO':
        audio = audio_sents(path_audio) # audio[phrase][mot] = string avec les valeurs (string d'un dictionnaire)
        #  par ex : "{u'moy_loc_B1': -0.059879489425627798, u'moy_loc_B2': -0.19947861555547755, u'moy_loc_F1': 0.026468}"
        mfcc = audio_sents(path_mfcc)
        text =  [None]*len(audio)
    elif opt == 'MULTI':
        audio = audio_sents(path_audio) # audio[phrase][mot] = string avec les valeurs (string d'un dictionnaire)
        #  par ex : "{u'moy_loc_B1': -0.059879489425627798, u'moy_loc_B2': -0.19947861555547755, u'moy_loc_F1': 0.026468}"
        mfcc = audio_sents(path_mfcc)
        
    X = [__sent2features(s, t, u, params) for (s, t, u) in zip(text, audio, mfcc)] # on prend phrase par phrase
    
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
    u"""Classe les labels par fréquence.
    Simple statistiques sur la frequence des labesls.    
    """
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
