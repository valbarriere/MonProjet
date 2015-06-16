# -*- coding: utf-8 -*-

from formatage import *
from extraction import *
import pycrfsuite

path = "C:/Users/claude-lagoutte/Documents/LUCAS/These/Python/Datasets/Semaine"

#%% Création des dump

dump_datasetsemaine(path+"/train")
dump_datasetsemaine(path+"/test")

#%% Extraction des données d'apprentissage

X_train, y_train = extract2CRFsuite(path+"/train/dump")

trainer = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train('basic_model')

#%% Tagging et comparaison sur la base de test

tagger = pycrfsuite.Tagger()
tagger.open('basic_model')