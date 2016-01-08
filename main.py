# -*- coding: utf-8 -*-
u"""main."""

from __future__ import division
from extraction import *
from mesures import *
import pycrfsuite
import os
from itertools import product

# path = "/home/lucasclaude3/Documents/Stage_Telecom/Datasets/Semaine/"
path = "/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/Datasets/Semaine/"
ALL_LABELS = {'attitude_positive', 'attitude_negative', 'source', 'target'}
ALL_FILES = sorted(os.listdir(path+"all/dump/")) # nom de tous les fichiers contenus dans path+"all/dump" tries dans l'ordre

#%%
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

# We just wanna test on the textual features
opt = 'TEXT'

params = {}
params['c1'] = 0
params['c2'] = 1e-2
params['max_it'] = 50
params['opt'] = opt 
params['context_negation'] = 2
params['nb_neighbours'] = 2

label = 'attitude'
label_select = 'attitude'

trainer = pycrfsuite.Trainer(verbose=False) # A voir sur pycrfsuite

for i in range(len(ALL_FILES)):  # i represente une session ?
    filename = ALL_FILES[i]
    X, y = extract2CRFsuite(path+"all/dump/"+filename,
                            path+"all/dump_audio/"+filename,
                            path+"all/dump_mfcc/"+filename,
                            label, params)
    for x_seq, y_seq in zip(X, y): # x_seq y_seq representent une phrase
        trainer.append(x_seq, y_seq,i) # (phrase,label_phrase,session)

#k = 0 # for possible hidden files
#for i in range(len(ALL_FILES)):  # i represente une session ?
#    filename = ALL_FILES[i]
#    if filename.split('.')[0] != '':
#        filename_model = filename.split('.')[0] # to throw away the extension
#        X, y = extract2CRFsuite(path+"all/dump/"+filename,
#                                path+"all/dump_audio/"+filename,
#                                path+"all/dump_mfcc/"+filename,
#                                label, opt)
#        for x_seq, y_seq in zip(X, y): # x_seq y_seq representent une phrase
#            trainer.append(x_seq, y_seq, k) # (phrase,label_phrase,session)
#        k+=1
#    else: # just a hidden file
#        i += 1
#        filename = ALL_FILES[i]

    
trainer.set_params({
    'c1': params['c1'],   # coefficient for L1 penalty
    'c2': params['c2'],  # coefficient for L2 penalty
    'max_iterations': params['max_it'],  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': False,
})

#%%
print("\n******\nBeginning of the training\n")

nb_test = 0
trainer.set_params({'c2' : 1e-2})

filename = ALL_FILES[nb_test] # On prend un dossier à mettre à l'écart pour la CV

filename_model = filename.split('.')[0] # to threw away the extension
path_model = '/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/MonProjet/'
print path_model + 'models/model_%s_' %opt +filename_model
trainer.train(path_model + 'models/model_%s_' %opt +filename_model, nb_test) # l'indice nb_test permet de s'entrainer sur tous les dossier sauf 1

X_test, y_test = extract2CRFsuite(path+"all/dump/"+filename,
                        path+"all/dump_audio/"+filename,
                        path+"all/dump_mfcc/"+filename,
                        label,params)
#%%
tagger = pycrfsuite.Tagger(verbose=False) # tagger sert a charger un modele qu'on va utiliser

tagger.open(path_model + 'models/model_%s_' %opt + filename_model) # tagger pour la session test, pour un dump ?

f = open(path_model+"current_dump_%s" %filename, 'w') # dump qui servira a avoir : le mot / son label VT / son label predit
truepos, falsepos, falseneg = (0, 0, 0)
for sent, corr_labels in zip(X_test, y_test): # pour faire par phrase ? cf extract2CRFsuite
    pred_labels = tagger.tag(sent)
    trueposAdd, falseposAdd, falsenegAdd = \
        F1_token(
            pred_labels,
            corr_labels,
            label_select) # renvoi le nombre de VP FP FN / PHRASE --> F1_token dans measures.py
    truepos += trueposAdd
    falsepos += falseposAdd
    falseneg += falsenegAdd
    txt_sent = ""
    for word in sent: # on remplit le current_dump ; current_dump_A... vieux trucs
        txt_sent += word['word']+"\t"
    f.write("\n"+txt_sent+"\n")
    f.write("\t".join(corr_labels)+"\n")
    f.write("\t".join(pred_labels)+"\n")

f.close()
        

# span overlap measure
precision = "%.2f" % (truepos/(truepos+falsepos+0.01) * 100) # 0.01 pour eviter div/0
recall= "%.2f" % (truepos/(truepos+falseneg+0.01) * 100)
print("\n******\nSpan overlap measure :\n")
print("Precision for label " + label + " : "
      + precision + " %")
print("Recall for label " + label + " : "
      + recall + " %")
print truepos, falsepos, falseneg

# nltk measure
#y_pred = [tagger.tag(x_seq) for x_seq in X_test]
#print("\n******\nNltk measure:\n")
#print(bio_classification_report(y_test, y_pred))


#%% Cross-validation


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
    print("Beginning of the training")
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


#cvloo('BIO','attitude_negative')
#cvloo('BIO','attitude_positive')

#cvloo('source') # ne pas oublier de changer le répertoire dump selon les cas
#cvloo('target')

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

label_att = ['attitude_negative','attitude_positive'] ; label_select = 'attitude_positive'
label_att = 'attitude_positive' ; label_select = 'attitude_positive' ; valence = True
label_att = 'attitude' ; label_select = None ; valence = False
cvloo(label_att, path_results, params, label_select = label_select, valence=valence)


#%%
LIST_C1 = [1e-5, 1e-6, 1e-7, 1e-8, 0]
LIST_C2 = [1e-2, 1e-3, 1e-1]
LIST_CONTXT_NEG = [0,1,2]
LOOP_TEST = True

for (c1,c2,context_negation) in product(LIST_C1,LIST_C2,LIST_CONTXT_NEG):
    params['c1'] = c1
    params['c2'] = c2
    params['context_negation'] = context_negation
    cvloo('attitude', path_results, params,LOOP_TEST=LOOP_TEST)
    
