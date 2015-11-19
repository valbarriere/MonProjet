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
# D_PATH = '/home/lucasclaude3/Documents/Stage_Telecom/'
D_PATH = "/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/"

""" Peut être le module en apparence le plus bordelique à cause de sa structure
hierarchique. Il permet d'extraire les features et les labels a partir des dump.
Pour bien comprendre ce qui se passe il faut partir de "exctrat2CRFsuite" et
remonter à chaque fois qu'il y a des appels de fonction.

Tu retrouveras aussi "count_labels" et "stats_labels" qui permettent de compter
le nombre d'occurences des labels et de produire les stats dessus."""


def read_patterns(path):
    u"""
    Utilise un fichier csv qui contient les expressions d'opinion contenues dans les 15 sessions qu'on utilise
    Lit les patterns détectés par les règles syntaxiques de Caro.
    Retourne un dict avec les patterns comme cles, et le sujet comme valeur
    Utilisation de tokenize de la toolbox nltk
    """
    dict_patterns = {}
    f = open(path, 'Ur')
    text = f.read()
    sents = text.split('\n') #separe par ligne
    for sent in sents:
        elements = sent.split(';') #separe par colonne du csv
        try:
            if elements[0] == 'session': # si on est sur une ligne qui sert a rien
                continue 
            # elements[0] seession ; elements[1] utterance id ?
            # elements[2] is the sent ; elements[3] affect/appreciation (grosse briques), 
            # elements[4] target of the expression; elements[5] polarity        
            
            sent = " ".join(nltk.word_tokenize(elements[2])) # separe le contractions (don't --> do n't)
            if not dict_patterns.__contains__(sent): # on met le sujet
                dict_patterns[sent] = nltk.word_tokenize(elements[4])
        except IndexError:
            print("End of patterns file")        
    return dict_patterns

PATH_PATTERN = '/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/MonProjet/patterns.csv'

PATTERNS = read_patterns(PATH_PATTERN) # On l'appelle ici pour le charger


def __merge_dicts(*dict_args):
    u"""Fusionne n'importe quel nombre de dict."""
    z = {}
    for y in dict_args:
        z.update(y)
    return z


def __word2features(sent, i, nb_neighbours):
    u"""Features lexicaux et syntaxiques.
    nb_neighbours est la taille du contexte que l'on prend en nombre de mots    
    """   

    word = sent[i][0].lower() # literallement le mot sans les maj
    postag = sent[i][1][:2] #pourquoi que les 2 premieres lettres du POS-tag uniquement ?
    features = {
        'bias': 1.0, # pourquoi ce bias ?
        'word': word,
        'postag': postag
    }   
    
    # Number of words that you take into the context
    if nb_neighbours == None:
        nb_neighbours = 2
    
    for k in range(1,nb_neighbours+1): # Begin at k = 1
        if i > k-1: # If not k-th word of the sentence
        
            word_neigh_buff = sent[i-k][0].lower()
            postag_buff = sent[i-k][1][:2]
            features.update({
                ('%d:word.lower=' %-k) : word_neigh_buff,
                ('%d:postag=' %-k) : postag_buff,
            })
        else: # If (k-1)-th word = Place In Sentence
            features[('P%dIS' %k)] = 1.0
        
        if i < len(sent) - k: # If not k-th last word of the sentence
            word_neigh_buff = sent[i+k][0].lower()
            postag[-k] = sent[i+k][1][:2]
            features.update({
                ('%d:word.lower=' %k) : word_neigh_buff,
                ('%d:postag=' %k) : postag_buff,
            })
        else: # If (len(sent) - k)-th word
            features[('P%dIS' %-k)] = 1.0
    
    boolVP = False
    for j in range(len(sent)):
        if sent[j][1][:2] == 'VB': # 2 premieres lettres du postag du mot j
            boolVP = True
            features['phrase_type']='VP'
    if boolVP is False:
        features['phrase_type']='NP'    
    # si ya VB ds la phrase VP, sinon NP
        
    ### SWN score ###
    try:
        # rappel : MORPHY_TAG = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
        tag_conversion = MORPHY_TAG[features['postag']]

        synset = list(swn.senti_synsets(features['word'], pos=tag_conversion))[0] # variable SWN assez long au niveau du tps
        # On choisit le 0         
        polarity = [synset.pos_score(), synset.neg_score(), synset.obj_score()] # score SWN (triplet)
        features.update({
            'synset.pos_score()': polarity[0],
            'synset.neg_score()': polarity[1],
            'synset.obj_score()': polarity[2]
        })
    except (KeyError, IndexError):
        pass
    
    return features


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


def __rules2features(sent, i):
    """
    result['inTarget'/'inRule'] = 1.0 si le mot i fait partie de la target ou non
    """
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
    
    Il n'y a qu'a fusionner les dict voulus pour chaque mot
    
    sent, audio et mfcc sont d'une seule phrase --> len(sent) est le nbr de mot i le numero du mot
    """
    return [__merge_dicts(__word2features(sent, i),
                          __audio2features(audio,i)) for i in range(len(sent))] 


def __sent2label(sent, label):
    return [__decision(str_labels, label) for token, postag,
            str_labels in sent]


def __sent2tokens(sent):
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
        if "I-"+label in list_labels:
            return "I-"+label
        elif "B-"+label in list_labels:
            return "B-"+label
        else:
            return "O"


def text_sents(path):
    u"""Traite le texte. Pas utilisé non ? """
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
                print('END OF FILE %s' % path[-3:]) #fin du fichier, donne le nom de la session en +
                break
        sents_list.append(words_list)
    return sents_list  


def extract2CRFsuite(path_text, path_audio, path_mfcc, label='BIO'):
    u"""PLUS IMPORTANTE.
    
    Extrait features et label pour une session
    à partir d'un dossier contenant les dump au format Conll
    """
    text = nltk.corpus.conll2002.iob_sents(path_text) # text[phrase][mot] = (mot, genre NN, BIO-attitude)
    audio = audio_sents(path_audio) # audio[phrase][mot] = string avec les valeurs (string d'un dictionnaire)
    #  par ex : "{u'moy_loc_B1': -0.059879489425627798, u'moy_loc_B2': -0.19947861555547755, u'moy_loc_F1': 0.026468}"
    mfcc = audio_sents(path_mfcc)       
    X = [__sent2features(s, t, u) for (s, t, u) in zip(text, audio, mfcc)] # on prend phrase par phrase
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
