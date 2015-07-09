# -*- coding: utf-8 -*-
u"""main."""

from __future__ import division
from formatage import *
from extraction import *
import pycrfsuite

ALL_LABELS = {'evaluation', 'affect', 'source', 'target'}
path = "/home/lucasclaude3/Documents/Stage_Telecom/Datasets/Semaine"

#%% Creation des dump

dump_datasetsemaine(path+"/train")
dump_datasetsemaine(path+"/test")

#%% Extraction des données d'apprentissage

for label in ALL_LABELS:
    X_train, y_train = extract2CRFsuite(path+"/train/dump", label)
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
neg = {}
for label in ALL_LABELS:
    tagger = pycrfsuite.Tagger(verbose=False)
    tagger.open('basic_model_'+label)

    X_test, y_test = extract2CRFsuite(path+"/test/dump", label)

    i = 0
    cpt = 0
    for sent, corr_labels in zip(X_test, y_test):
        pred_labels = tagger.tag(sent)
        if pred_labels == corr_labels:
            i += 1
        else:
            neg[cpt] = 1
        cpt += 1

    well_tagged[label] = str(i/cpt * 100)
    print("Well-tagged sentences for label " + label + " : "
          + well_tagged[label] + " %")

well_tagged['global'] = str((cpt-len(neg))/cpt * 100)
print("Well-tagged sentences : "+well_tagged['global']+" %")

f = open('bin_classifiers_results_exact', 'w')
for lab in well_tagged.keys():
    f.write(lab+"\t"+well_tagged[lab])
    f.write('\n')

f.close()

#%% Tagging overlap measure

well_tagged = {}
y_pred = {}
y_corr = {}
for label in ALL_LABELS:
    tagger = pycrfsuite.Tagger(verbose=False)
    tagger.open('basic_model_'+label)
    X_test, y_test = extract2CRFsuite(path+"/test/dump", label)
    cpt = 0
    nb_words = 0
    nb_pos = 0
    for sent, corr_labels in zip(X_test, y_test):
        pred_labels = tagger.tag(sent)
        for j in range(len(corr_labels)):
            if pred_labels[j] == corr_labels[j]:
                nb_pos += 1
            if (cpt, j) not in y_pred.keys():
                y_pred[cpt, j] = [pred_labels[j]]
                y_corr[cpt, j] = [corr_labels[j]]
            else:
                y_pred[cpt, j].append(pred_labels[j])
        cpt += 1
        nb_words += len(corr_labels)
    well_tagged[label] = str(nb_pos/nb_words * 100)
    print("Well-tagged tokens for label " + label + " : "
          + well_tagged[label] + " %")

nb_pos = 0
for key in y_pred.keys():
    if set(y_pred[key]) == set(y_corr[key]):
        nb_pos += 1
well_tagged['global'] = str(nb_pos/len(y_pred.keys()) * 100)
print("Well-tagged tokens : " + well_tagged['global'] + " %")

f = open('bin_classifiers_results_overlap', 'w')
for lab in well_tagged.keys():
    f.write(lab+"\t"+well_tagged[lab])
    f.write('\n')

f.close()
