# -*- coding: utf-8 -*-

from formatage import *
from extraction import *
import nltk
import os

path = "C:/Users/claude-lagoutte/Documents/LUCAS/These/Python/Datasets/Semaine"

#%% Création des dump

dump_datasetsemaine(path+"/train")
dump_datasetsemaine(path+"/test")

#%% Extraction des données d'apprentissage
 
X_train, y_train = extract2CRFsuite(path+"/train/dump")