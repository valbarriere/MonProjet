# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:37:07 2015

@author: Valou
"""

from nltk import word_tokenize
from nltk.corpus import sentiwordnet as swn
from sys import platform
from numpy import sign


# To know if I am on the MAC or on the PC with Linux             
CURRENT_OS = platform   
if CURRENT_OS == 'darwin':         
    INIT_PATH = "/Users/Valou/"
elif CURRENT_OS == 'linux2':
    INIT_PATH = "/home/valentin/"

PATH_PATTERN = INIT_PATH + 'Dropbox/TELECOM_PARISTECH/Stage_Lucas/MonProjet/patterns.csv'


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
            #print("End of patterns file")
            pass
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
    postag = sent[i][1][:2] #2 premieres lettres du POS-tag uniquement car decrit plus simplement
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

MORPHY_TAG = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r', 'WR': 'r'}
def __swn_scores(features):
    """
    The SentiWordNet scores of each word
    Redundance since the POS is contained in the SWN score, and we give it afterward
    """
    try:
        # rappel : MORPHY_TAG = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r', 'WR': 'r'}
        tag_conversion = MORPHY_TAG[features['postag']]
    
        synset = list(swn.senti_synsets(features['word'], pos=tag_conversion))[0] # variable SWN assez long au niveau du tps
        # On choisit le 0         
        polarity = [synset.pos_score(), synset.neg_score(), synset.obj_score()] # score SWN (triplet)
        features.update({
            'sentisynset.pos': polarity[0],
            'sentisynset.neg': polarity[1],
            'sentisynset.obj': polarity[2]
        })
#        return polarity
    except (KeyError, IndexError):
#        return (None,None,None)
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


PATH_LEXICON = INIT_PATH + 'Dropbox/TELECOM_PARISTECH/Corpus-Lexiques/Lexiques/'

def return_lexicon(PATH):
    f = open(PATH)
    var = f.readlines()
    f.close()
    lex = []
    for i in range(len(var)):
        lex.append(var[i][:-1])
    return lex
    

NEGATION_TOKENS = PATH_LEXICON + 'Minqing_Hu/negation-tokens.txt'  
list_nega = return_lexicon(NEGATION_TOKENS)
list_nega_0 = ["n't",'not','no']

def __negation(sent, i, context):
    """
    If there is a negation at least context word before the word
    ---------->WORK<----------
    """
    bool_neg = False
    for k in range(1,context+1): # Begin at k = 1
        if i > k-1 and ( sent[i-k][0].lower() in list_nega_0) : # If not k-th word of the sentence
            bool_neg = True
    
    return bool_neg
    
def __newI(sent,i):
    """
    Return True if there is a "and" (pos = CC) then a I
    
    ---------->DON'T WORK<----------
    """
    if i > 0 and sent[i-1][1][:2] == "CC" and sent[i][0] =="i":
        return True
    else:
        return False

################################ New features ############################
def __adj(sent,i):
    """
    Return True if the word is a noun preceded by an adj
    """
    if i > 0 and sent[i-1][1][:2] == "JJ" and sent[i][1][:2] =="NN":
        return True
    else:
        return False    

def __adv(sent,i):
    """
    Return True if the preceding word is an adverb other than not
    """
    if i > 0 and sent[i-1][1][:2] == "RB" and sent[i-1][0].lower() !="not":
        return True
    else:
        return False    


######## Features de Lexicon-based approach #############

import pickle
SO_CAL_NAME = PATH_LEXICON + 'SO-CAL/SO-CAL_Lexicon.PKL'

# lex [POS+word]=score ; POS : 876 'adv', 2820 'adj', 1539 'noun', 1130 'verb' 217 'int' 
lex = pickle.load(open(SO_CAL_NAME,'rb'))
lex_keys = lex.keys()
# the keys are the beginning POS-tag contained in Caro's dumps
MORPHY_TAG_LEX = {'NN': 'noun', 'JJ': 'adj', 'VB': 'verb', 'RB': 'adv', 'WR': 'adv'}

# On aurait : if MORPHY_TAG_LEX[POS]+word in lex.keys() le mot aurait une valeur d'opinion
list_intens = {}
for key in lex_keys:
    if key[:3] == 'int': list_intens[key[3:]]=lex[key]
del list_intens[''] # petit bug a corriger

def __intens(sent,i):
    """
    Return True if the preceding word is an intensifier
    """
    if i > 0 and sent[i-1][0].lower() in list_intens:
        return True
    else:
        return False    
def __intens_is(sent,i):
    """
    Return True if the word itself is an intensifier
    """
    if sent[i][0].lower() in list_intens:
        return True
    else:
        return False 


def __SO_value(sent,i):
    """
    Return the SO value of the word if it has one, False either
    """
    word = sent[i][0].lower() # literallement le mot sans les maj
    postag = sent[i][1][:2] #2 premieres lettres du POS-tag uniquement car decrit plus simplement
    
    if postag+word in lex_keys:
        return lex[postag+word]
    else: 
        return False

# Add the new features in the list of you add it in params
LIST_FEATURES = ['nb_neighbours','context_negation','rules_synt', 'newI', \
'rules_synt', 'swn_scores','swn_pos', 'swn_neg', 'swn_obj', 'inverse_score', \
'adj', 'adv','intens','intens_is','SO_value','SO_intensifier','SO_negation']

def __sent2features(sent,params):
    """
    Cette fonction sera utilisé plus tard lorsqu'on aura de toute la structure de la phrase
    pour faire marcher differentes regles
    """
    # Creation of the variables
    feat = {}
    for i in range(len(sent)):
        feat[i] = {}
    
    for i in range(len(sent)):
        if __SO_value(sent,i):
            if __intens_is(sent,i):
                sent

    return feat

def __word2features(sent, i, params):
    u"""Features lexicaux et syntaxiques.
    nb_neighbours est la taille du contexte que l'on prend en nombre de mots
    ---->Si l'on veut ajouter une feature, il faut la mettre dans params et ensuite 
    l'ajouter dans la liste LIST_FEATURES pour la prendre en compte si elle y est   
    """      
    boolean = {}
    
    for k in LIST_FEATURES:
        if k in params:
            boolean[k] = params[k]
        else:
            boolean[k] = 0
      
    # Basic features of each word : those ones are always there by default
    features = __features_base(sent, i, boolean['nb_neighbours'])
    
    # si ya VB ds la phrase VP, sinon NP
    features['phrase_type'] = __phrase_type(sent) 
    
    ### SWN score ###
    if boolean['swn_scores'] != 0:
        features = __swn_scores(features)
        
    if 'sentisynset.pos' in features.keys(): 
        bool_senti = True
    else: 
        bool_senti = False 

    ### More SWN score ####
    if boolean['swn_pos'] != 0 and bool_senti:
        features['swn_pos_score_bis'] = features['sentisynset.pos']    
    if boolean['swn_neg'] != 0 and bool_senti:
        features['swn_pos_score_bis'] = features['sentisynset.neg']            
    if boolean['swn_obj'] != 0 and bool_senti:
        features['swn_obj_score_bis'] = features['sentisynset.obj']      
    
    if boolean['context_negation'] != 0:
        features['negation'] = __negation(sent, i, boolean['context_negation'])
    
    # True if there is a 'I' after a 'and'
    if boolean['newI'] == True:
        features['newI'] = __newI(sent,i)
    
    # rules_syntaxic
    if boolean['rules_synt'] == True:
        features = __rules2features(features, sent, i)
        
    # If negation before, inverse the pos and neg score
    if boolean['inverse_score'] == True and features['negation'] == True and bool_senti:
        buff = features['sentisynset.pos']        
        features['sentisynset.pos'] = features['sentisynset.neg']
        features['sentisynset.neg'] = buff
    
    if boolean['adj'] == True:
        features['adj'] = __adj(sent,i)
        
    if boolean['adv'] == True:
        features['adv'] = __adv(sent,i)

    if boolean['intens'] == True:
        features['intens'] = __intens(sent,i)
        
    if boolean['intens_is'] == True:
        features['intens_is'] = __intens_is(sent,i)        
    
        
    if boolean['SO_value'] == True:
        SO_value = __SO_value(sent,i)
        # If there is a value
        if SO_value:
            # if the word is before intensifier
            if boolean['SO_intensifier']:
                if __intens(sent,i) and SO_value:
                    SO_value = (1+list_intens[sent[i].lower()])*SO_value
                #if __intens_is(sent,i) and __SO_value(sent,i+1):
                #    SO_val = (1+list_intens[sent[i].lower()])*__SO_value(sent,i+1)
            
            if boolean['SO_negation']:        
                if __negation(sent, i, boolean['SO_negation']):
                    SO_value = SO_value - 4*sign(SO_value)
            
            features['SO_value'] = SO_value
    
    
    return features 
    
    
