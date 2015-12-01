# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:35:09 2015

@author: Valou
"""
from main import cvloo

params = {}
params['max_it'] = 50
params['opt'] = 'TEXT' 

path_results = '/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/MonProjet/results/'
params['c2'] = 1e-3
params['c1'] = 0
params['context_negation'] = 2
params['newI'] = False
params['nb_neighbours'] = 2
params['rules_synt'] = False

label_att = ['attitude_negative','attitude_positive'] ; label_select = 'attitude_positive'
label_att = 'attitude_positive' ; label_select = 'attitude_positive' ; valence = True
label_att = 'attitude' ; label_select = None ; valence = False
cvloo(label_att, path_results, params, label_select = label_select, valence=valence)

