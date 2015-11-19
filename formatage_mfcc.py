# -*- coding: utf-8 -*-
"""
Formatage mfcc
"""
from scikits.audiolab import Sndfile, Format
from features import mfcc
import numpy as np
import glob
import os

""" Pareil que "formatage_audio" mais pour les MFCC.
Dans l'absolu tu peux essayer de fusionner les 2 pour plus de clareté."""

#S_PATH = '/home/lucasclaude3/Documents/Stage_Telecom/Datasets/Semaine/Sessions/'
#D_PATH = '/home/lucasclaude3/Documents/Stage_Telecom/MonProjet/'

S_PATH = "/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/Datasets/Semaine/Sessions/"
D_PATH = "/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/MonProjet/"

#%% First step : preprocessing

f = Sndfile(D_PATH+"tests_audio/wavtest.wav")

n = int(f.nframes)
fs = f.samplerate 
nc = f.channels # checker que c'est bien 1
enc = f.encoding # checker que c'est bien 'pcm24'

data = f.read_frames(n, np.float32)
m = np.mean(data, dtype=np.float32)
v = np.var(data, dtype=np.float32)
data = (data - m) / np.sqrt(v)

# optionnel, juste pour les tests:
data = data[:400000] # premier tour de parole session 25 Spike operator :150000

f_normalized = Sndfile("tests_audio/wavtest_normalized.wav", 'w', Format('wav', enc), 1, fs)
f_normalized.write_frames(data)
f_normalized.close()

#%% Second step : processing

mel = mfcc(data, samplerate=fs, winlen=0.04, winstep=0.02)
mel_min = np.amin(mel)
mel_max = np.amax(mel)

#%% compute mean and var for preprocessing

moy = {}
var = {}
for i in range(1, mel.shape[1]):
    moy["mfcc_%s" %i] = np.mean(map(lambda x: mel[x,i], mel[:,i]))
    var["mfcc_%s" %i] = np.var(map(lambda x: mel[x,i], mel[:,i]))
    print("%s --> moy = %.2f and var = %.2f" % (i, moy["mfcc_%s" %i], var["mfcc_%s" %i]))

#%% Third step : implement read_turn

def read_turn(mel, moy, var, turn): 
    
    mel_min = np.amin(mel)    
    mel_max = np.amax(mel)
    
    text = ""
    for line in turn:
        if line == '':
            continue
        
        features = {}
        words = line.split()
        start = float(words[0]) / 1000
        end = float(words[1]) / 1000
        
        time_scale = [t*0.02 for t in range(mel.shape[0])]
        times = [t/0.02 for t in time_scale if (t>start and t<end)]
        
        for i in range(1, mel.shape[1]):
            j = "mfcc_%s" %i
#            features["moy_loc_%s" %j] =\
#                np.mean(map(lambda x: (mel[x,i] - moy[j]) / np.sqrt(var[j]), times))
#            features["var_loc_%s" %j] =\
#                np.var(map(lambda x: (mel[x,i] - moy[j]) / np.sqrt(var[j]), times))
            
            features["moy_loc_%s" %j] =\
                np.mean(map(lambda x: (mel[x,i] - mel_min) / (mel_max - mel_min), times))
            features["var_loc_%s" %j] =\
                np.var(map(lambda x: (mel[x,i] - mel_min) / (mel_max - mel_min), times))               
        
        try:
            text += words[2]+'\t'+str(features)+'\n'
        except BaseException, e:
            print e
    
    return text
    
f = open('transcript_test', 'Ur')
text = f.read()
turns = text.split('\n.\n')
turn = turns[1]
turn_formated = turn.split('\n')[1:]
dump = read_turn(mel, moy, var, turn_formated)

#%% Fourth step : read a file

def normalize_signal(path1, path2):
    """attention, retourne directement le tableau du signal."""
    f = Sndfile(path1)
    
    n = int(f.nframes)
    fs = f.samplerate 
    nc = f.channels # checker que c'est bien 1
    if nc != 1:
        raise Exception('Fichier wav possédant plusieurs canaux.')
    enc = f.encoding # checker que c'est bien 'pcm24'
    if enc != 'pcm24':
        raise Exception("Encodage différent de 'pcm24'.")
    
    data = f.read_frames(n, np.float64)
    m = np.mean(data, dtype=np.float64)
    v = np.var(data, dtype=np.float64)
    data = (data - m) / np.sqrt(v)
    
    return data, fs


def read_file(session_dir):
    
    name = session_dir
    if len(name) == 1:
        name2 = '00'+name
    elif len(name) == 2:
        name2 = '0'+name
    else:
        name2 = name

    print "Chargement des différents fichiers... session %s" % session_dir
    # trouver les fichiers audio
    l_op = glob.glob(S_PATH+name+'/*Operator HeadMounted*.wav')
    l_us = glob.glob(S_PATH+name+'/*User HeadMounted*.wav')
    if len(l_op) != 1 or len(l_us) != 1:
        raise Exception('Zero ou multiples matchs pour les fichiers audio')
    else:
        wav_op = l_op[0]
        wav_us = l_us[0]

    # trouver les fichiers txt:
    l_op = glob.glob(S_PATH+name+'/word*operator*')
    l_us = glob.glob(S_PATH+name+'/word*user*')
    if len(l_op) != 1 or len(l_us) != 1:
        raise Exception('Zero ou multiples matchs pour les fichiers txt')
    else:
        txt_op = l_op[0]
        txt_us = l_us[0]
    
    # normaliser le signal
    try:
        data_op, fs_op = normalize_signal(wav_op, S_PATH+"normalized/op_"+name)
        data_us, fs_us = normalize_signal(wav_us, S_PATH+"normalized/us_"+name)
    except BaseException, e:
        print e 
    
    # scripts mfcc operator
    print "Calcul des mfcc operator..."
    mel_op = mfcc(data_op, samplerate=fs_op, winlen=0.04, winstep=0.02)
       
    moy_op = {}
    var_op = {}
    for i in range(1, mel_op.shape[1]): # remove first coefficient
        j = "mfcc_%s" %i
        moy_op[j] = np.mean(map(lambda x: mel_op[x,i], mel_op[:,i]))
        var_op[j] = np.var(map(lambda x: mel_op[x,i], mel_op[:,i]))
        
    # scripts mfcc user
    print "Calcul des mfcc user..."
    mel_us = mfcc(data_us, samplerate=fs_us, winlen=0.04, winstep=0.02)
       
    moy_us = {}
    var_us = {}
    for i in range(1, mel_us.shape[1]): # remove first coefficient
        j = "mfcc_%s" %i
        moy_us[j] = np.mean(map(lambda x: mel_us[x,i], mel_us[:,i]))
        var_us[j] = np.var(map(lambda x: mel_us[x,i], mel_us[:,i]))
        
    # chargement des fichiers
    try:
        f_op = open(txt_op,'Ur')
        turns_op = f_op.read().split('\n.\n')
        f_op.close()
        
        f_us = open(txt_us,'Ur')
        turns_us = f_us.read().split('\n.\n')
        f_us.close()
    except NameError:
        print('Text file %s not found' % name)    
    
    
    print "Lecture des tours de parole..."
    cpt_op = 0
    cpt_us = 0
    complete_dump = ""
    while True:
        #print cpt_op
        #print cpt_us
        turn_op = turns_op[cpt_op].split('\n')[1:]
        turn_us = turns_us[cpt_us].split('\n')[1:]
            
        try:
            start_op = float(turn_op[0].split()[0])
        except IndexError:
            start_op = float('Inf')
            
        try:
            start_us = float(turn_us[0].split()[0])
        except IndexError:
            start_us = float('Inf')
        try:
            if start_op < start_us:
                dump = read_turn(mel_op, moy_op, var_op, turn_op)
                cpt_op += 1
            else:
                dump = read_turn(mel_us, moy_us, var_us, turn_us)
                cpt_us += 1
        except BaseException, e:
            print e
            print turn_op
            print turn_us
        complete_dump += "\n\n"+dump
        if cpt_op >= len(turns_op)-1 and cpt_us >= len(turns_us)-1:
            break
    
    f2 = open(D_PATH+'/dump_'+name2, 'w')
    f2.write(complete_dump)
    f2.close()
    
#read_file('25')

#%% read all files

for session_dir in sorted(os.listdir(S_PATH)):
    if session_dir == "normalized":
        continue
    else:
        read_file(session_dir)