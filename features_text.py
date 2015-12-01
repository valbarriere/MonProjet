# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:37:07 2015

@author: Valou
"""

from nltk import word_tokenize
from nltk.corpus import sentiwordnet as swn

MORPHY_TAG = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
PATH_PATTERN = '/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/MonProjet/patterns.csv'


def read_patterns(path):
    u"""
    Utilise un fichier csv qui contient les expressions d'opinion contenues dans les 15 sessions qu'on utilise
    Lit les patterns détectés par les règles syntaxiques de Caro.
    RETOURNE un dict avec les patterns comme cles, et le sujet comme valeur
    
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
            
            sent = " ".join(word_tokenize(elements[2])) # separe le contractions (don't --> do n't)
            if not dict_patterns.__contains__(sent): # on met le sujet
                dict_patterns[sent] = word_tokenize(elements[4])
        except IndexError:
            print("End of patterns file")        
    return dict_patterns

PATTERNS = read_patterns(PATH_PATTERN) # On l'appelle ici pour le charger


def __rules2features(features, sent, i):
    """
    features['inRule'] = 1.0 si la phrase est un pattern detecté par caro
    features['inTarget'] = 1.0 si mot i est target du pattern  
    """
    
    formated_sent = " ".join([sent[k][0] for k in range(len(sent))])
    if formated_sent in PATTERNS:
        features['inRule'] = 1.0
        target = PATTERNS[formated_sent]
        if sent[i][0] in target:
            features['inTarget'] = 1.0
    return features
    
    
def __features_base(sent, i, nb_neighbours):
    """
    Basic features of each word, including the word and pos-tags of the context
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
            postag_buff = sent[i+k][1][:2]
            features.update({
                ('%d:word.lower=' %k) : word_neigh_buff,
                ('%d:postag=' %k) : postag_buff,
            })
        else: # If (len(sent) - k)-th word doesn't exist
            features[('P%dIS' %-k)] = 1.0

    return features            

def __swn_scores(features):
    """
    The SentiWordNet scores of each word
    """
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

def __swn_scores2(features):
    """
    The SentiWordNet scores of each word
    """
    try:
        # rappel : MORPHY_TAG = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
        tag_conversion = MORPHY_TAG[features['postag']]
    
        synset = list(swn.senti_synsets(features['word'], pos=tag_conversion))[0] # variable SWN assez long au niveau du tps
        # On choisit le 0         
        polarity = [synset.pos_score(), synset.neg_score(), synset.obj_score()] # score SWN (triplet)
        features.update({
            'synset.pos_score()2': polarity[0],
            'synset.neg_score()2': polarity[1],
            'synset.obj_score()2': polarity[2]
        })
    except (KeyError, IndexError):
        pass
    
    return features
    

    
def __phrase_type(sent):
    """
    Return VB if there is a verbe in the sentence
    """
    boolVP = False
    for j in range(len(sent)):
        if sent[j][1][:2] == 'VB': # 2 premieres lettres du postag du mot j
            boolVP = True
            return 'VP'
    if boolVP is False:
        return 'NP'  
    
def __negation(sent, i, context):
    """
    If there is a negation at least context word before the word
    """
    bool_neg = False
    for k in range(1,context+1): # Begin at k = 1
        if i > k-1 and sent[i-k][0].lower() == ("not" or "n't") : # If not k-th word of the sentence
            bool_neg = True
    
    return bool_neg
    
def __newI(sent,i):
    """
    Return True if there is a "and" then a I
    """
    if i > 0 and sent[i-1][1][:2] == "CC" and sent[i][0] =="i":
        return True
    else:
        return False


def __word2features(sent, i, params):
    u"""Features lexicaux et syntaxiques.
    nb_neighbours est la taille du contexte que l'on prend en nombre de mots    
    """   
    
    # Add the new features in the list of you add it in params
    LIST_FEATURES = ['nb_neighbours','context_negation','rules_synt', 'newI', \
    'rules_synt']
    boolean = {}
    
    for k in LIST_FEATURES:
        if k in params:
            boolean[k] = params[k]
        else:
            boolean[k] = 0
      
    # Basic features of each word
    features = __features_base(sent, i, boolean['nb_neighbours'])
    
    # si ya VB ds la phrase VP, sinon NP
    features['phrase_type'] = __phrase_type(sent) 
    
    ### SWN score ###        
    features = __swn_scores(features)
    #features = __swn_scores2(features)
    
    if boolean['context_negation'] != 0:
        features['negation'] = __negation(sent, i, boolean['context_negation'])
    
    # True if there is a 'I' after a 'and'
    if boolean['newI'] == True:
        features['newI'] = __newI(sent,i)
    
    # rules_syntaxic
    if boolean['rules_synt'] == True:
        features = __rules2features(features, sent, i)
    
    
    return features 
    
    
