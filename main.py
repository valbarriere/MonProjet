# -*- coding: utf-8 -*-
u"""main."""

from __future__ import division
from formatage import *
from extraction import *
from mesures import *
import pycrfsuite
import os

ALL_LABELS = {'attitude', 'source', 'target'}
ALL_FILES = os.listdir(path+"/all/aa1")
path = "/home/lucasclaude3/Documents/Stage_Telecom/Datasets/Semaine"

#%% Creation des dump

dump_datasetsemaine(path+"/train")
dump_datasetsemaine(path+"/test")
dump_datasetsemaine(path+"/all")

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

#%% Tagging F1-measure w/ span overlap comparison

precision = {}
recall = {}
for label in ALL_LABELS:
    truepos, trueneg, falsepos, falseneg = (0, 0, 0, 0)
    tagger = pycrfsuite.Tagger(verbose=False)
    tagger.open('basic_model_'+label)
    X_test, y_test = extract2CRFsuite(path+"/test/dump", label)
    for sent, corr_labels in zip(X_test, y_test):
        pred_labels = tagger.tag(sent)
        trueposAdd, truenegAdd, falseposAdd, falsenegAdd = \
            F1_span_overlap(
                pred_labels,
                corr_labels,
                label)
        truepos += trueposAdd
        trueneg += truenegAdd
        falsepos += falseposAdd
        falseneg += falsenegAdd
    print(truepos, trueneg, falsepos, falseneg)
    precision[label] = str(truepos/(truepos+falsepos) * 100)
    print("Precision for label " + label + " : "
          + precision[label] + " %")
    recall[label] = str(truepos/(truepos+falseneg) * 100)
    print("Recall for label " + label + " : "
          + recall[label] + " %")

f = open('bin_classifiers_results_F1', 'w')
for lab in precision.keys():
    f.write(lab+"\t"+precision[lab]+"\t"+recall[lab])
    f.write('\n')
f.close()

#%% Cross-validation : en cours


def cv(label):
    u"""Compute the Cross-validation for the given label."""
    precision = {}
    recall = {}
    for filename in ALL_FILES:
        # créer un dossier test et un dossier train temporaires et répartir
        shutil.remove(path+"/all/dump/train_temp", True)
        shutil.remove(path+"/all/dump/test_temp", True)
        for filename2 in ALL_FILES:
            if filename2 != filename:
                shutil.move(path+"/all/dump/"+filename,
                            path+"/all/dump/train_temp/"+filename)
            else:
                shutil.move(path+"/all/dump/"+filename,
                            path+"/all/dump/test_temp/"+filename)
        Xtrain, ytrain = extract2CRFsuite(path+"/all/dump/train_temp", label)
        Xtest, ytest = extract2CRFsuite(path+"/all/dump/test_temp", label)
        # train
        train_step(Xtrain, ytrain, 'model_'+label, label)
        # test
        precision[filename], recall[filename] = test_step(Xtest,
                                                          ytest,
                                                          'model_'+label,
                                                          label)
        # compute statistics on precision and recall


def train_step(X_train, y_train, model_name, label):
    u"""Compute the train step of CV."""
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
    trainer.train('model_'+label)


def test_step(X_test, y_test, model_name, label):
    u"""Compute the test step of CV."""
    truepos, trueneg, falsepos, falseneg = (0, 0, 0, 0)
    tagger = pycrfsuite.Tagger(verbose=False)
    tagger.open(model_name)
    for sent, corr_labels in zip(X_test, y_test):
        pred_labels = tagger.tag(sent)
        trueposAdd, truenegAdd, falseposAdd, falsenegAdd = \
            F1_span_overlap(
                pred_labels,
                corr_labels,
                label)
        truepos += trueposAdd
        trueneg += truenegAdd
        falsepos += falseposAdd
        falseneg += falsenegAdd
    precision = str(truepos/(truepos+falsepos) * 100)
    recall = str(truepos/(truepos+falseneg) * 100)
    return precision, recall
