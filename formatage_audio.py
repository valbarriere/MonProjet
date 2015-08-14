# -*- coding: utf-8 -*-
u"""Méthodes pour formater l'audio.

A faire tourner en python 2.
note : dans semaine il manque le transcript complet pour la session 100
"""

from scikits.audiolab import Sndfile, Format
from praatinterface import PraatLoader
import math
import os
import glob
import re

def read_turn(opened_file, first_line):
    new_line = first_line
    not_finished = True
    while "." not in "".join(new_line):
        print("".join(new_line.split('\n'))) # truc à changer
        new_line = opened_file.readline()
        if new_line == '':
            not_finished = False
            break
    opened_file.readline()
    return opened_file.readline(), not_finished
        
#%%

path = '/home/lucasclaude3/Documents/Stage_Telecom/Datasets/Semaine/Sessions/'

cpt = 0
for session_dir in os.listdir(path):
    cpt+=1
    name = session_dir

    # trouver les fichiers audio    
    l_op = glob.glob(path+name+'/*Operator HeadMounted*.wav')
    l_us = glob.glob(path+name+'/*User HeadMounted*.wav')
    if len(l_op) != 1 or len(l_us) != 1:
        raise Exception('Zero ou multiples matchs pour les fichiers audio')
    else:
        wav_op = "".join(l_op)
        wav_us = "".join(l_us)

    # trouver les fichiers txt:
    l_op = glob.glob(path+name+'/word*operator*')
    l_us = glob.glob(path+name+'/word*user*')
    # l_transcript = glob.glob(path+name+'/alignedTranscript*')
    # transcript non exploitables car il y a des manques dans SEMAINE
    
    if len(l_op) != 1 or len(l_us) != 1:
        raise Exception('Zero ou multiples matchs pour les fichiers txt')
    # elif len(l_transcript) != 1:
        # print('Dossier %s surement incomplet' % name)
    else:
        txt_op = "".join(l_op)
        txt_us = "".join(l_us)
        # txt_transcripts = "".join(l_transcript)
    
    if len(name) == 1:
        name = '00'+name
    elif len(name) == 2:
        name = '0'+name
    else:
        pass
    
    # chargement des fichiers
    try:
        turns_op = open(txt_op,'r')
        turns_us = open(txt_us,'r')
        # turns_transcript = open(txt_transcripts,'r')
    except NameError, e:
        print('File %s not found' % name)
    
    m_op = None
    m_us = None
    while m_op is None:
        turn_op = "".join(turns_op.readline().split())
        m_op = re.search('recording.*turn', turn_op)       
    
    while m_us is None:
        turn_us = "".join(turns_us.readline().split())
        m_us = re.search('recording.*turn', turn_us)
        
   
    turn_op = turns_op.readline()
    turn_us = turns_us.readline()
    
    start_op = 0
    start_us = 0
    
    while min(start_op, start_us) < float('Inf'):
        try:
            start_op = int(turn_op.split()[0])
        except IndexError:
            start_op = float('Inf')
        try:
            start_us = int(turn_us.split()[0])
        except IndexError:
            start_us = float('Inf')
        if start_op < start_us:
            turn_op, bool1 = read_turn(turns_op, turn_op)
            print("******")
        else:
            turn_us, bool2 = read_turn(turns_us, turn_us)
            print("******")

    if cpt == 2:
        break
        
        
        
        
    

        
    
    
    
    
