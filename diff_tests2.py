# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:35:09 2015

@author: Valou
"""
import os
from extraction import extract2CRFsuite
path = "/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/Datasets/Semaine/"
ALL_LABELS = {'attitude_positive', 'attitude_negative', 'source', 'target'}
ALL_FILES = sorted(os.listdir(path+"all/dump/")) # nom de tous les fichiers contenus dans path+"all/dump" tries dans l'ordre

label = 'attitude'
label_select = 'attitude'

for i in range(1):  # i represente une session ?
    filename = ALL_FILES[i]
    X, y = extract2CRFsuite(path+"all/dump/"+filename,
                            path+"all/dump_audio/"+filename,
                            path+"all/dump_mfcc/"+filename,
                            label, 'TEXT')
                            