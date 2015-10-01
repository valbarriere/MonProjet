# -*- coding: utf-8 -*-
u"""main."""

from __future__ import division
from extraction import *
from mesures import *
import pycrfsuite
import os

path = "/home/lucasclaude3/Documents/Stage_Telecom/Datasets/Semaine/"
ALL_LABELS = {'attitude_positive', 'attitude_negative', 'source', 'target'}
ALL_FILES = sorted(os.listdir(path+"all/dump/"))


#%% Un seul hold out pour tester le code

label = 'attitude'
label_select = 'attitude'

trainer = pycrfsuite.Trainer(verbose=False)

for i in range(len(ALL_FILES)):
    filename = ALL_FILES[i]
    X, y = extract2CRFsuite(path+"all/dump/"+filename,
                            path+"all/dump_audio/"+filename,
                            path+"all/dump_mfcc/"+filename,
                            label)
    for x_seq, y_seq in zip(X, y):
        trainer.append(x_seq, y_seq, i)
    
trainer.set_params({
    'c1': 0,   # coefficient for L1 penalty
    'c2': 1e-2,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': False,
})
print("\n******\nBeginning of the training\n")

filename = ALL_FILES[1]
trainer.train('models/model_'+filename, 1)
X_test, y_test = extract2CRFsuite(path+"all/dump/"+filename,
                        path+"all/dump_audio/"+filename,
                        path+"all/dump_mfcc/"+filename,
                        label)
tagger = pycrfsuite.Tagger(verbose=False)
tagger.open('models/model_'+filename)

f = open("current_dump_%s" %filename, 'w')
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
    txt_sent = ""
    for word in sent:
        txt_sent += word['word']+"\t"
    f.write("\n"+txt_sent+"\n")
    f.write("\t".join(corr_labels)+"\n")
    f.write("\t".join(pred_labels)+"\n")

f.close()
        

# span overlap measure
precision = "%.2f" % (truepos/(truepos+falsepos+0.01) * 100)
recall= "%.2f" % (truepos/(truepos+falseneg+0.01) * 100)
print("\n******\nSpan overlap measure :\n")
print("Precision for label " + label + " : "
      + precision + " %")
print("Recall for label " + label + " : "
      + recall + " %")


# nltk measure
#y_pred = [tagger.tag(x_seq) for x_seq in X_test]
#print("\n******\nNltk measure:\n")
#print(bio_classification_report(y_test, y_pred))


#%% Cross-validation


def dump_resultats(precision, recall, filename):
    u"""Dump the results."""
    f = open(filename, 'w')
    for session in precision.keys():
        f.write("%s\t%s\t%s\n" % (session, precision[session], recall[session]))
    f.close()


def cvloo(label, label_select=None):
    u"""Compute the Cross-validation for the given label."""
    if label_select is None:
        label_select = label
    
    truepos_o, falsepos_o, falseneg_o = (0, 0, 0)    
    precision = {}
    recall = {}
    
    trainer = pycrfsuite.Trainer(verbose=False)

    for i in range(len(ALL_FILES)):
        filename = ALL_FILES[i]
        X, y = extract2CRFsuite(path+"all/dump/"+filename,
                                path+"all/dump_audio/"+filename,
                                path+"all/dump_mfcc/"+filename,
                                label)
        for x_seq, y_seq in zip(X, y):
            trainer.append(x_seq, y_seq, i)
        
    trainer.set_params({
        'c1': 0,   # coefficient for L1 penalty
        'c2': 1e-2,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': False,
    })
    print("Beginning of the training")
    for i in range(len(ALL_FILES)):
        filename = ALL_FILES[i]
        trainer.train('models/model_'+filename, i)
        X_test, y_test = extract2CRFsuite(path+"all/dump/"+filename,
                                path+"all/dump_audio/"+filename,
                                path+"all/dump_mfcc/"+filename,
                                label)
        tagger = pycrfsuite.Tagger(verbose=False)
        tagger.open('models/model_'+filename)
        
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
        
    dump_resultats(precision, recall, 'results_CVLOO_audio_'+label+"_"+label_select)
    return precision, recall


#cvloo('BIO','attitude_negative')
#cvloo('BIO','attitude_positive')

cvloo('attitude') # ne pas oublier de changer le répertoire dump selon les cas
