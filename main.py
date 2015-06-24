# -*- coding: utf-8 -*-

from __future__ import division
from formatage import *
from extraction import *
import pycrfsuite
import cPickle as pickle

ALL_LABELS = {'evaluation','affect','source','cible'}
path = "C:/Users/claude-lagoutte/Documents/LUCAS/These/Python/Datasets/Semaine"

#%% Création des dump

dump_datasetsemaine(path+"/train")
dump_datasetsemaine(path+"/test")

#%% Extraction des données d'apprentissage

for label in ALL_LABELS:
    X_train, y_train = extract2CRFsuite(path+"/train/dump",label)    
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)    
    trainer.set_params({
        'c1': 0,   # coefficient for L1 penalty
        'c2': 1e-2,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
    
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })    
    trainer.train('basic_model_'+label)

#%% Tagging exact measure

well_tagged = {}
pos = [1]*len(X_test)
for label in ALL_LABELS:
    tagger = pycrfsuite.Tagger(verbose=False)
    tagger.open('basic_model_'+label)
    
    X_test, y_test = extract2CRFsuite(path+"/test/dump",label)
    
    i=0
    cpt=0
    for sent, corr_labels in zip(X_test,y_test):
        pred_labels = tagger.tag(sent)
        if pred_labels == corr_labels:
            i+=1
        else:
            pos[cpt] = 0
        cpt+=1
        
    well_tagged[label] = str(i/len(X_test) * 100)
    print("Well-tagged sentences for label "+label+" : "+well_tagged[label]+" %")
    
well_tagged['global'] = str(np.sum(pos)/len(X_test) * 100)
print("Well-tagged sentences : "+well_tagged['global']+"%")

f = open('bin_classifiers_results_exact','w')
for lab in well_tagged.keys():
    f.write(lab+"\t"+well_tagged[lab])
    f.write('\n')

f.close()

#%% Tagging overlap measure

well_tagged = {}
for label in ALL_LABELS:
    tagger = pycrfsuite.Tagger(verbose=False)
    tagger.open('basic_model_'+label)
    
    X_test, y_test = extract2CRFsuite(path+"/test/dump",label)
    
    i=0
    for sent, corr_labels in zip(X_test,y_test):
        pred_labels = tagger.tag(sent)
        
        n=0
        for j in range(len(corr_labels)):
            if pred_labels[j] == corr_labels[j]:
                n+=1
        i+= n/len(corr_labels)
        
    well_tagged[label] = str(i/len(X_test) * 100)
    print("Well-tagged sentences for label "+label+" : "+well_tagged[label]+" %")
    
f = open('bin_classifiers_results_overlap','w')
for lab in well_tagged.keys():
    f.write(lab+"\t"+well_tagged[lab])
    f.write('\n')

f.close()