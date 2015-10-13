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

"""
Le module qui permet de lancer les XP !
En gros on importe extraction qui te permet d'extraire les features 
et les labels a partir des fichiers dump.
Puis on lance l'apprentissage avec CRFsuite, suivi du test.

Un premier bloc pour tester le code ci-dessous, sur une session uniquement.
Un deuxième pour lancer la cross-validation complete.

REMARQUES SUPER IMPORTANTES :

******
1) label et label_select, kezako ??!

Alors comme je te l'avais explique, chaque mot peut posseder plusieurs labels
simultanement. ex: B-source; I-attitude. La première approche que j'avais 
utilisee au début du stage consistait à ignorer cet aspect multilabel, 
en disant par exemple que source l'emporte sur attitude. Dans l'ex précédent 
on aurait donc garde uniquement B-source. Supposons maintenant que tu as appris
ton modèle, que tu t'en sers pour tagger tes phrases test. Pour comparer la 
performance, tu as besoin de raisonner uniquement sur UN type de label, c'est 
à dire de filtrer en post-processing. C'est la raison d'etre du "label_select",
qui correspond au label que tu selectionnes quant tu compares la prediction et
la verite terrain dans F1_token. Quant à label, ici il vaudra 'BIO', 
pour signifier que l'on garde toutes les annotations pour l'apprentissage.

Le bon cote de cette méthode c'est qu'on peut apprendre un seul modele pour tous
les labels. Le mauvais c'est qu'en faisant ça on perd de l'information, et en
plus il est probable qu'on coupe certains segments en plein milieu. ex : 
un B-source tout seul en plein milieu d'un segment attitude.
Donc c'est pas terrible. D'autant que les stats sur le dataset (cf mon rapport)
montrent que le multi-label arrive dans 10% des cas.

Donc il faut une approche multilabel. Pour ce faire, le plus simple est d'
apprendre un classifieur pour chaque type de span (source, target, attitude).
Et donc dans cette optique, il faut extraire uniquement les annotations qui 
nous intéressent dans le dump. Dans ce cas, label = label_select = TRUC.
Normalement, tu n'auras besoin que de cette méthode.

Si tu veux te servir de 'BIO' fais bien attention, les definitions ne 
sont surement plus à jour dans extract2CRFsuite. 

******
2) lorsque tu sélectionnes les dump pour le texte en invoquant 
extract2CRFsuite, fais gaffe il y a plusieurs dossiers
en fonction de ce que tu cherches à faire:
- DUMP_ALL avec toutes les annotations (source, target, attitudes detaillees...)
- DUMP_ATTITUDEPOSNEGONLY
- DUMP
Regarde bien celui que tu dois utiliser en fonction de la tache
"""


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

#cvloo('source') # ne pas oublier de changer le répertoire dump selon les cas
#cvloo('target')

cvloo('attitude')