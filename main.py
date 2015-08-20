# -*- coding: utf-8 -*-
u"""main."""

from __future__ import division
from formatage import *
from extraction import *
from mesures import *
import pycrfsuite
import os
import shutil

path = "/home/lucasclaude3/Documents/Stage_Telecom/Datasets/Semaine"
ALL_LABELS = {'attitude', 'source', 'target'}
ALL_FILES = os.listdir(path+"/all/aa1")

#%% Creation des dump


dump_datasetsemaine(path+"/train")
dump_datasetsemaine(path+"/test")
dump_datasetsemaine(path+"/all")
count_labels(path+"/all/dump", "labels_occurrences")

#%% Extraction des données d'apprentissage


for label in ALL_LABELS:
    X_train, y_train = extract2CRFsuite(path+"/train/dump", label)
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
        'c1': 0,  # coefficient for L1 penalty
        'c2': 1e-2,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train('models/basic_model_'+label)

#%% Tagging F1-measure w/ span overlap comparison


precision = {}
recall = {}
for label in ALL_LABELS:
    truepos, falsepos, falseneg = (0, 0, 0)
    tagger = pycrfsuite.Tagger(verbose=False)
    tagger.open('models/basic_model_'+label)
    X_test, y_test = extract2CRFsuite(path+"/test/dump", label)
    for sent, corr_labels in zip(X_test, y_test):
        pred_labels = tagger.tag(sent)
        trueposAdd, falseposAdd, falsenegAdd = \
            F1_span_overlap(
                pred_labels,
                corr_labels,
                label)
        truepos += trueposAdd
        falsepos += falseposAdd
        falseneg += falsenegAdd
    precision[label] = str(truepos/(truepos+falsepos+0.01) * 100)
    print("\n******")
    print("Precision for label " + label + " : "
          + precision[label] + " %")
    recall[label] = str(truepos/(truepos+falseneg+0.01) * 100)
    print("Recall for label " + label + " : "
          + recall[label] + " %")
    y_pred = [tagger.tag(x_seq) for x_seq in X_test]
    print(bio_classification_report(y_test, y_pred))

f = open('results/bin_classifiers_results_F1_continuous', 'w')
for lab in precision.keys():
    f.write(lab+"\t"+precision[lab]+"\t"+recall[lab])
    f.write('\n')
f.close()


#%% Cross-validation


def cvloo(label, label_select=None):
    u"""Compute the Cross-validation for the given label."""
    if label_select is None:
        label_select = label
    precision = {}
    recall = {}
    truepos, falsepos, falseneg = (0, 0, 0)
    for filename in ALL_FILES:
        # créer un dossier test et un dossier train temporaires et répartir
        shutil.rmtree(path+"/all/dump/train_temp", True)
        shutil.rmtree(path+"/all/dump/test_temp", True)
        os.mkdir(path+"/all/dump/train_temp")
        os.mkdir(path+"/all/dump/test_temp")
        for filename2 in ALL_FILES:
            if filename2 != filename:
                shutil.copy(path+"/all/dump/dump_"+filename2[:-3],
                            path+"/all/dump/train_temp/dump_"+filename2[:-3])
            else:
                shutil.copy(path+"/all/dump/dump_"+filename2[:-3],
                            path+"/all/dump/test_temp/dump_"+filename2[:-3])
        Xtrain, ytrain = extract2CRFsuite(path+"/all/dump/train_temp", label)
        Xtest, ytest = extract2CRFsuite(path+"/all/dump/test_temp", label)
        # train
        train_step(Xtrain, ytrain, 'models/model_'+label)
        # test
        precision[filename[:-3]], recall[filename[:-3]],\
            trueposAdd, falseposAdd, falsenegAdd = test_step(
                Xtest, ytest, 'models/model_'+label, label_select, bool_info)
        truepos += trueposAdd
        falsepos += falseposAdd
        falseneg += falsenegAdd
    precision['overall'] = "%.2f" % (truepos/(truepos+falsepos+0.01) * 100)
    recall['overall'] = "%.2f" % (truepos/(truepos+falseneg+0.01) * 100)
    dump_resultats(precision, recall, 'results_CVLOO_continuous_'+label+"_"+label_select)


def train_step(X_train, y_train, model_name):
    u"""Compute the train step of CV."""
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
        'c1': 0,   # coefficient for L1 penalty
        'c2': 1e-2,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': False,
    })
    trainer.train(model_name)


def test_step(X_test, y_test, model_name, label):
    u"""Compute the test step of CV."""
    truepos, falsepos, falseneg = (0, 0, 0)
    tagger = pycrfsuite.Tagger(verbose=False)
    tagger.open(model_name)
    for sent, corr_labels in zip(X_test, y_test):
        pred_labels = tagger.tag(sent)
        trueposAdd, falseposAdd, falsenegAdd = \
            F1_span_overlap(
                pred_labels,
                corr_labels,
                label)
        truepos += trueposAdd
        falsepos += falseposAdd
        falseneg += falsenegAdd
    precision = "%.2f" % (truepos/(truepos+falsepos+0.01) * 100)
    recall = "%.2f" % (truepos/(truepos+falseneg+0.01) * 100)
    return precision, recall, truepos, falsepos, falseneg


def dump_resultats(precision, recall, filename):
    u"""Dump the results."""
    f = open(filename, 'w')
    for session in precision.keys():
        f.write("%s\t%s\t%s\n" % (session, precision[session], recall[session]))
    f.close()


cvloo('BIO', 'source')
cvloo('BIO', 'target')
cvloo('BIO', 'attitude')
cvloo('source')
cvloo('target')
cvloo('attitude')
