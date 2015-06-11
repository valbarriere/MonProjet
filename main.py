# -*- coding: utf-8 -*-

from formatage import *
from extraction import *
import nltk
import os

path = "C:/Users/claude-lagoutte/Documents/LUCAS/These/Python/Datasets/Semaine"
dump_semaine(path+"/ac1/session025.ac",path+"/aa1/session025.aa",'dumptest')
train_sents = nltk.corpus.conll2002.iob_sents(os.getcwd()+"/dumptest")
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2label(s,"evaluation") for s in train_sents]