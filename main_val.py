# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:26:12 2015

@author: Valou
"""

from __future__ import division
from extraction import extract2CRFsuite
from mesures import F1_token
import pycrfsuite
import os
from itertools import product
import numpy as np

# path = "/home/lucasclaude3/Documents/Stage_Telecom/Datasets/Semaine/"
path = "/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/Datasets/Semaine/"
ALL_LABELS = {'attitude_positive', 'attitude_negative', 'source', 'target'}
ALL_FILES = sorted(os.listdir(path+"all/dump/")) # nom de tous les fichiers contenus dans path+"all/dump" tries dans l'ordre

def dump_resultats(precision, recall, F1, filename):
    u"""Dump the results."""
    f = open(filename, 'w')
    f.write("Session\t\tPrecision\tRecall\tF1\n")
    
    session = 'overall' # Every sessions on the 1st line, then
    f.write("%s\t\t%s\t\t%s\t\t%.2f\n" % (session, precision[session], recall[session], F1))
    alz = precision.copy()
    del alz['overall']
    for session in alz.keys():
        f.write("%s\t%s\t\t%s\n" % (session, precision[session], recall[session]))
        
    f.close()

def dump_resultats_total(precision, recall, F1, filename, params):
    """
    Dump the results in ONE text file with all the parameters
    """    
    f = open(filename, 'ab')
    k=1
    for p in params:
        f.write("%s : " %p + str(params[p]) + "\t" + int(np.floor((3-np.mod(k,3))/3))*"\n")
        k+=1
    f.write("\nSession\t\tPrecision\tRecall\t\tF1\n")
    session = 'overall' # Stats over all sessions on the 1st line, then
    f.write("%s\t\t%s\t\t%s\t\t%.2f\n\n" % (session, precision[session], recall[session], F1))
    alz = precision.copy()
    del alz['overall']
    for session in alz.keys():
        f.write("%s\t%s\t\t%s\n" % (session, precision[session], recall[session]))
        
    f.write("******************************************************************************\n\n")        
    f.close()
        
#%%
def cvloo(label, path_results, params, label_select=None, LOOP_TEST=False, valence = False):
    u"""Compute the Cross-validation for the given label.
    valence is True if we wanna distinguish the positive and negative attitudes    
    """
    if label_select is None:
        label_select = label
    opt = params['opt']
    
    truepos_o, falsepos_o, falseneg_o = (0, 0, 0)    
    precision = {}
    recall = {}
    
    trainer = pycrfsuite.Trainer(verbose=False)
        
    
    for i in range(len(ALL_FILES)):
        filename = ALL_FILES[i]
        X, y = extract2CRFsuite(path+"all/dump"+valence*"_attitudeposneg_only"+"/"+filename,
                                path+"all/dump_audio/"+filename,
                                path+"all/dump_mfcc/"+filename,
                                label, params)
        for x_seq, y_seq in zip(X, y):
            trainer.append(x_seq, y_seq, i)
        
    trainer.set_params({
        'c1': params['c1'],   # coefficient for L1 penalty
        'c2': params['c2'],  # coefficient for L2 penalty
        'max_iterations': params['max_it'],  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': False,
    })
    #print("Beginning of the training")
    for i in range(len(ALL_FILES)):
    #for i in range(1):
        
        filename = ALL_FILES[i]
        filename_model = filename.split('.')[0] # to threw away the extension
        path_model = '/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/MonProjet/'
        
        # Training 
        trainer.train(path_model+'models/model_%s_' %opt + filename_model, i)

        # Testing
        X_test, y_test = extract2CRFsuite(path+"all/dump"+valence*"_attitudeposneg_only"+"/"+filename,
                                path+"all/dump_audio/"+filename,
                                path+"all/dump_mfcc/"+filename,
                                label, params)
        tagger = pycrfsuite.Tagger(verbose=False)
        tagger.open(path_model+'models/model_%s_' %opt + filename_model)
        
        truepos, falsepos, falseneg = (0, 0, 0)
        for sent, corr_labels in zip(X_test, y_test):
            pred_labels = tagger.tag(sent)
            trueposAdd, falseposAdd, falsenegAdd = \
                F1_token(
                    pred_labels,
                    corr_labels,
                    label_select)
            truepos += trueposAdd
            falsepos += falseposAdd
            falseneg += falsenegAdd
        
        precision[filename] = "%.2f" % (truepos/(truepos+falsepos+0.01) * 100)
        recall[filename] = "%.2f" % (truepos/(truepos+falseneg+0.01) * 100)            
        truepos_o += truepos
        falsepos_o += falsepos
        falseneg_o += falseneg
        
    precision['overall'] = "%.2f" % (truepos_o/(truepos_o+falsepos_o+0.01) * 100)
    recall['overall'] = "%.2f" % (truepos_o/(truepos_o+falseneg_o+0.01) * 100)
    F1 = 2*float(precision['overall'])*float(recall['overall'])/(float(precision['overall'])+float(recall['overall'])+1e-5)

    # If there is pos and neg differentiation for the attitudes
    if valence == True and label.__class__ == list: label = 'attitud_posneg'

    ext = '.txt'
    dump_resultats(precision, recall, F1, path_results + 'results_CVLOO_%s_' %(opt) +label+"_"+label_select+ext)
    if LOOP_TEST: # if loop test dump the ALL the results in 1 file
        dump_resultats_total(precision, recall, F1, path_results + 'results_total_%s_' %(opt) +label+"_"+label_select+ext, params)
    return_sent = 'Precision : %s, Recall : %s, F1 : %.2f' %(precision['overall'], recall['overall'], F1)
    return return_sent


#%%
params = {}
params['c1'] = 0
params['c2'] = 1e-2
params['max_it'] = 50
params['opt'] = 'TEXT' 
params['context_negation'] = 2
params['nb_neighbours'] = 2
params['newI'] = False
params['swn_scores'] = True


#%%
path_results = '/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/MonProjet/results/'
params['c2'] = 1e-3
params['c1'] = 0
params['context_negation'] = 2
params['nb_neighbours'] = 2
params['rules_synt'] = False
params['swn_score'] = True
params['inverse_score'] = False

params['swn_pos'] = True
params['swn_neg'] = True
#params['swn_obj'] = False

label_att = ['attitude_negative','attitude_positive'] ; label_select = 'attitude_positive' ; valence = True
#label_att = 'attitude_positive' ; label_select = 'attitude_positive' ; valence = True
#label_att = 'attitude' ; label_select = None ; valence = False
print cvloo(label_att, path_results, params, label_select = label_select, valence=valence)



#%%
LOOP_TEST = False
if LOOP_TEST:
    LIST_C1 = [1e-5, 1e-6, 1e-7, 1e-8, 0]
    LIST_C2 = [1e-2, 1e-3, 1e-1]
    LIST_CONTXT_NEG = [0,1,2]
    
    for (c1,c2,context_negation) in product(LIST_C1,LIST_C2,LIST_CONTXT_NEG):
        params['c1'] = c1
        params['c2'] = c2
        params['context_negation'] = context_negation
        cvloo('attitude', path_results, params,LOOP_TEST=LOOP_TEST)
    
