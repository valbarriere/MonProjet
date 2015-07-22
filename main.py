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
    precision[label] = str(truepos/(truepos+falsepos+0.01) * 100)
    print("Precision for label " + label + " : "
          + precision[label] + " %")
    recall[label] = str(truepos/(truepos+falseneg+0.01) * 100)
    print("Recall for label " + label + " : "
          + recall[label] + " %")

f = open('bin_classifiers_results_F1', 'w')
for lab in precision.keys():
    f.write(lab+"\t"+precision[lab]+"\t"+recall[lab])
    f.write('\n')
f.close()

#%% Cross-validation : en cours


def cvloo(label):
    u"""Compute the Cross-validation for the given label."""
    precision = {}
    recall = {}
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
        train_step(Xtrain, ytrain, 'model_'+label, label)
        # test
        precision[filename[:-3]], recall[filename[:-3]] = test_step(
            Xtest, ytest, 'model_'+label, label)
    dump_resultats(precision, recall, 'results_CVLOO_'+label)
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
        # print(pred_labels)
        # print(corr_labels)
        # print("******")
        trueposAdd, truenegAdd, falseposAdd, falsenegAdd = \
            F1_span_overlap(
                pred_labels,
                corr_labels,
                label)
        truepos += trueposAdd
        trueneg += truenegAdd
        falsepos += falseposAdd
        falseneg += falsenegAdd
    precision = "%.2f" % (truepos/(truepos+falsepos+0.01) * 100)
    recall = "%.2f" % (truepos/(truepos+falseneg+0.01) * 100)
    return precision, recall


def dump_resultats(precision, recall, filename):
    u"""Dump the results."""
    f = open(filename, 'w')
    for session in precision.keys():
        f.write("%s\t%s\t%s\n" % (session, precision[session], recall[session]))
    prec_values = list(precision.values())
    rec_values = list(recall.values())
    for i in range(len(prec_values)):
        prec_values[i] = float(prec_values[i])
        rec_values[i] = float(rec_values[i])
    f.write("mean\t%.2f\t%.2f\n" % (np.mean(prec_values), np.mean(rec_values)))
    f.write("std\t%.2f\t%.2f\n" % (np.std(prec_values), np.std(rec_values)))
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
        if eval(multi_lab) != {'O'}:
            total_woO += int(cpt)
        if len(eval(multi_lab)) not in dict_nb:
            dict_nb[len(eval(multi_lab))] = 0
            if eval(multi_lab) != {'O'}:
                dict_nb[len(eval(multi_lab))] += int(cpt)
        else:
            dict_nb[len(eval(multi_lab))] += int(cpt)
    f.close()
    ranking = sorted(occurrences, key=lambda data: data[1], reverse=True)
    f = open(stats_filename, 'w')
    f.write("labels\toccurrences\tfréquences\n")
    for i in range(len(ranking)):
        f.write("%s\t%d\t%f\n" % (ranking[i][0],
                                  ranking[i][1],
                                  ranking[i][1]/total_wO * 100))
    f.write("\nnb_labels\toccurrences\tfréquences\n")
    for key in sorted(dict_nb):
        f.write("%s\t%d\t%f\n" % (key,
                                  dict_nb[key],
                                  dict_nb[key]/total_woO * 100))
    f.close()
