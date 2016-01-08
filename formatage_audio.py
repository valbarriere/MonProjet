# -*- coding: utf-8 -*-
"""
Formatage audio
"""
from scikits.audiolab import Sndfile, Format
from praatinterface import PraatLoader
import numpy as np
import glob
import os

#S_PATH = '/home/lucasclaude3/Documents/Stage_Telecom/Datasets/Semaine/Sessions/'
#D_PATH = '/home/lucasclaude3/Documents/Stage_Telecom/MonProjet/'

S_PATH = "/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/Datasets/Semaine/Sessions/"
D_PATH = "/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/MonProjet/"

""" Le module qui pond les dump audio.
C'est un peu galere à expliquer mais si tu regardes les noms des blocs tu
devrais comprendre.

On calcule les features audio tous les 40ms avec un script PRAAT, puis on les pre-process. 
Dans "read_turn" on lit les dates de debut et de fin de chaque mot, et on 
prend moyenne et variance sur la durée correspondante. 
Au sein de "read_file" il faut ensuite charger les fichiers wav de chaque 
interlocuteur, et lire les tours de parole DANS LE BON ORDRE !

ATTENTION !!! 
Un probleme qui revient parfois est que les temps de début de chaque mot sont 
parfois mal encodes, ce qui produit des differences dans l'ordre des tours 
entre le texte et l'audio. Donc parfois il faut retoucher les fichiers dumps à 
la main en inversant des tours de parole."""

#%% First step : preprocessing

# Test pour reenregistrer le wav sous un autre format plus simple a traiter

f = Sndfile(D_PATH+"tests_audio/wavtest.wav") # ouvre 

n = int(f.nframes)
fs = f.samplerate 
nc = f.channels # checker que c'est bien 1
enc = f.encoding # checker que c'est bien 'pcm24'

data = f.read_frames(n, np.float32)
m = np.mean(data, dtype=np.float32)
v = np.var(data, dtype=np.float32)
data = (data - m) / np.sqrt(v)

# optionnel, juste pour les tests: juste 8secondes
data = data[:150000] # premier tour de parole session 25 Spike operator :150000

f_normalized = Sndfile(D_PATH+"tests_audio/wavtest_normalized.wav", 'w', Format('wav', enc), 1, fs)
f_normalized.write_frames(data)
f_normalized.close()

#%% Second step : processing (Praat) --> formants+pitch+intensite


# subprocess.call(['/Applications/Praat.app/Contents/MacOS/Praat', '--run', 
# '/Applications/Praat.app/Contents/MacOS/praatScripts/formants.praat', D_PATH+'tests_audio/wavtest_normalized.wav', '5', '5500'])

#P_PATH  = "/Applications/Praat.app/Contents/MacOS/"
P_PATH = "/Applications/Praat.app"
pl = PraatLoader(praatpath=D_PATH+'Praat.app')

text_formants = pl.run_script('formants.praat', D_PATH+'tests_audio/wavtest_normalized.wav', 5, 5500)
formants = pl.read_praat_out(text_formants)

text_pitch = pl.run_script('pitch.praat', D_PATH+'tests_audio/wavtest_normalized.wav')
pitch = pl.read_praat_out(text_pitch)

text_intensity = pl.run_script('intensity.praat', D_PATH+'tests_audio/wavtest_normalized.wav')
intensity = pl.read_praat_out(text_intensity)

#%% compute mean and var for preprocessing

moy = {}
var = {}
for i in [u'B1', u'B2', u'F1', u'F2']: # F1 : formant1 B1 ??
    moy[i] = np.mean(map(lambda x: formants[x][i], formants))
    var[i] = np.var(map(lambda x: formants[x][i], formants))
    print("%s --> moy = %.2f and var = %.2f" % (i, moy[i], var[i]))
    
    # y != 0 pour ne prendre que les moments ou l'on a du pitch
i = u'Pitch'
moy[i] = np.mean(filter(lambda y: y!=0, map(lambda x: pitch[x][i], pitch)))
var[i] = np.var(filter(lambda y: y!=0, map(lambda x: pitch[x][i], pitch)))
print("%s --> moy = %.2f and var = %.2f" % (i, moy[i], var[i]))

i = u'Intensity(dB)'
moy[i] = np.mean(map(lambda x: intensity[x][i], intensity))
var[i] = np.var(map(lambda x: intensity[x][i], intensity))
print("%s --> moy = %.2f and var = %.2f" % (i, moy[i], var[i]))

#%% Third step : implement read_turn

def read_turn(formants, pitch, intensity, turn): 
    """
    On lit les dates de debut et de fin de chaque mot, et on 
    prend moyenne et variance sur la durée correspondante.
    On effectue une normalisation Z pour chaque composante (X-mu)/sigma
    """
    
#    moy = {}
#    var = {}
#    for i in [u'B1', u'B2', u'F1', u'F2']: # F1 : formant1 B1 ??
#        moy[i] = np.mean(map(lambda x: formants[x][i], formants))
#        var[i] = np.var(map(lambda x: formants[x][i], formants))
#        
#    # y != 0 pour ne prendre que les moments ou l'on a du pitch
#    i = u'Pitch'
#    moy[i] = np.mean(filter(lambda y: y!=0, map(lambda x: pitch[x][i], pitch)))
#    var[i] = np.var(filter(lambda y: y!=0, map(lambda x: pitch[x][i], pitch)))
#    
#    i = u'Intensity(dB)'
#    moy[i] = np.mean(map(lambda x: intensity[x][i], intensity))
#    var[i] = np.var(map(lambda x: intensity[x][i], intensity))


    features = {}
    text = ""
    for line in turn:
        if line == '':
            continue
        
        words = line.split()
        start = float(words[0]) / 1000
        end = float(words[1]) / 1000
        
#        # formants
#        # times represente donc la duree d'un mot ! 
#        times = [t for t in formants if (t>start and t<end)]
#        for i in [u'B1', u'B2', u'F1', u'F2']:
#            features["moy_loc_%s" % i] =\
#                np.mean(map(lambda x: (formants[x][i] - moy[i]) / np.sqrt(var[i]), times))
#            features["var_loc_%s"% i] =\
#                np.var(map(lambda x: (formants[x][i] - moy[i]) / np.sqrt(var[i]), times))
#
#        # pitch
#        times = [t for t in pitch if (t>start and t<end)]
#        for i in [u'Pitch']:
#            forbidden = - moy[i] / np.sqrt(var[i])
#            features["moy_loc_%s" % i] =\
#                np.mean(filter(lambda y: y!=forbidden, map(lambda x: (pitch[x][i] - moy[i]) / np.sqrt(var[i]), times)))
#            features["var_loc_%s"% i] =\
#                np.var(filter(lambda y: y!=forbidden, map(lambda x: (pitch[x][i] - moy[i]) / np.sqrt(var[i]), times)))
#        
#        # intensity
#        times = [t for t in intensity if (t>start and t<end)]
#        for i in [u'Intensity(dB)']:
#            features["moy_loc_%s" % i] =\
#                np.mean(map(lambda x: (intensity[x][i] - moy[i]) / np.sqrt(var[i]), times))
#            features["var_loc_%s"% i] =\
#                np.var(map(lambda x: (intensity[x][i] - moy[i]) / np.sqrt(var[i]), times))

        # formants
        # times represente donc la duree d'un mot ! 
        times = [t for t in formants if (t>start and t<end)]
        for i in [u'B1', u'B2', u'F1', u'F2']:
            features["moy_loc_%s" % i] =\
                np.mean(map(lambda x: formants[x][i], times))
            features["var_loc_%s"% i] =\
                np.var(map(lambda x: formants[x][i] , times))

        # pitch
        times = [t for t in pitch if (t>start and t<end)]
        for i in [u'Pitch']:
            forbidden = 0 # since sometimes there is no pitch
            features["moy_loc_%s" % i] =\
                np.mean(filter(lambda y: y!=forbidden, map(lambda x: pitch[x][i] , times)))
            features["var_loc_%s"% i] =\
                np.var(filter(lambda y: y!=forbidden, map(lambda x: pitch[x][i] , times)))
        
        # intensity
        times = [t for t in intensity if (t>start and t<end)]
        for i in [u'Intensity(dB)']:
            features["moy_loc_%s" % i] =\
                np.mean(map(lambda x: intensity[x][i], times))
            features["var_loc_%s"% i] =\
                np.var(map(lambda x: intensity[x][i], times))        
        
        try:
            text += words[2]+'\t'+str(features)+'\n'
        except BaseException, e:
            print e

    return text
    
f = open('transcript_test', 'Ur') # Ca sort d'ou ce truc ?
text = f.read()
turns = text.split('\n.\n')
turn = turns[0]
turn_formated = turn.split('\n')[1:]
dump = read_turn(formants, pitch, intensity, turn_formated)

#%% Fourth step : read a file

def normalize_signal(path1, path2):
    """ 
    Prend le fichier audio dans path1, le normalise et le cree dans path2
    """
    
    f = Sndfile(path1)
    
    n = int(f.nframes)
    fs = f.samplerate 
    nc = f.channels 
    
    if nc != 1: # checker que nc est bien 1
        raise Exception('Fichier wav possédant plusieurs canaux.')
        
    enc = f.encoding
    if enc != 'pcm24': # checker que enc est bien 'pcm24'
        raise Exception("Encodage différent de 'pcm24'.")
    
    data = f.read_frames(n, np.float64)
    m = np.mean(data, dtype=np.float64)
    v = np.var(data, dtype=np.float64)
    
    # normalization
    data = (data - m) / np.sqrt(v)
    
    # On met ca dans path2
    f_normalized = Sndfile(path2, 'w', Format('wav', enc), 1, fs)
    f_normalized.write_frames(data)
    f_normalized.close()


def read_file(session_dir, NORMALIZED = False):
    """
    On calcule les features audio qui sont faits toutes les 20ms et et on les
    moyennes sur la durée des mots
    """

    name = session_dir 
    
    # Si on normalise les fichiers d'origine, sinon on prend ceux deja normalisés (+ simple)
    if NORMALIZED:
        print "Chargement des différents fichiers..."
        # trouver les fichiers audio : separation op et user
        # glob permet de recup tous les fichiers avec une expression dans le titre
        l_op = glob.glob(S_PATH+name+'/*Operator HeadMounted*.wav') 
        l_us = glob.glob(S_PATH+name+'/*User HeadMounted*.wav')
        if len(l_op) != 1 or len(l_us) != 1:
            raise Exception('Zero ou multiples matchs pour les fichiers audio')
        else:
            wav_op = l_op[0]
            wav_us = l_us[0]
    
        # trouver les fichiers txt: -->
        l_op = glob.glob(S_PATH+name+'/word*operator*')
        l_us = glob.glob(S_PATH+name+'/word*user*')
        if len(l_op) != 1 or len(l_us) != 1:
            raise Exception('Zero ou multiples matchs pour les fichiers txt')
        else:
            txt_op = l_op[0]
            txt_us = l_us[0]
        
        # normaliser le signal et le met dans le folder normalized
        try:
            normalize_signal(wav_op, S_PATH+"normalized/op_"+name)
            normalize_signal(wav_us, S_PATH+"normalized/us_"+name)
        except BaseException, e:
            print e

    pl = PraatLoader(praatpath=D_PATH+'praat')   
 
#################################### AUDIO ####################################
   
    # Scripts praat OPERATOR
    print "Lancement des scripts praat operator, puis user" # Cree une VARIABLE praat a base de fichiers .wav
    
    formants = {}
    pitch = {}
    intensity = {}
    moy = {}
    var = {}
    for spkr in ["op", "us"]:  
        text_formants = pl.run_script('formants.praat', S_PATH+"normalized/"+spkr+"_"+name, 5, 5500)
#        formants_op = pl.read_praat_out(text_formants)
        formants[spkr] = pl.read_praat_out(text_formants)
                
        text_pitch = pl.run_script('pitch.praat', S_PATH+"normalized/"+spkr+"_"+name)
#        pitch_op = pl.read_praat_out(text_pitch)
        pitch[spkr] = pl.read_praat_out(text_pitch)
        
        text_intensity = pl.run_script('intensity.praat', S_PATH+"normalized/"+spkr+"_"+name)
#        intensity_op = pl.read_praat_out(text_intensity)
        intensity[spkr] = pl.read_praat_out(text_intensity)
        

###################        
        # Taking the mean and the var of each feature for each speaker 
        for i in [u'B1', u'B2', u'F1', u'F2']:
            moy[spkr][i] = np.mean(map(lambda x: formants[spkr][x][i], formants[spkr]))
            var[spkr][i] = np.var(map(lambda x: formants[spkr][x][i], formants[spkr]))
            
            # Normalization
            for x in formants[spkr].keys():
                formants[spkr][x][i] = (formants[spkr][x][i] - moy[spkr][i])/np.sqrt(var[spkr][i])
        
        # y != 0 pour ne prendre que les moments ou l'on a du pitch
        i = u'Pitch'
        moy[spkr][i] = np.mean(filter(lambda y: y!=0, map(lambda x: pitch[spkr][x][i], pitch[spkr])))
        var[spkr][i] = np.var(filter(lambda y: y!=0, map(lambda x: pitch[spkr][x][i], pitch[spkr])))

        # Normalization
        for x in pitch[spkr].keys():
            pitch[spkr][x][i] = (pitch[spkr][x][i] - moy[spkr][i])/np.sqrt(var[spkr][i])
        
        i = u'Intensity(dB)'
        moy[spkr][i] = np.mean(map(lambda x: intensity[spkr][x][i], intensity[spkr]))
        var[spkr][i] = np.var(map(lambda x: intensity[spkr][x][i], intensity[spkr]))
 
        # Normalization
        for x in intensity[spkr].keys():
            intensity[spkr][x][i] = (intensity[spkr][x][i] - moy[spkr][i])/np.sqrt(var[spkr][i])
            
############

            
            
    
#    moy_op = {}
#    var_op = {}
#    for i in [u'B1', u'B2', u'F1', u'F2']:
#        moy_op[i] = np.mean(map(lambda x: formants_op[x][i], formants_op))
#        var_op[i] = np.var(map(lambda x: formants_op[x][i], formants_op))
#        
#    i = u'Pitch'
#    moy_op[i] = np.mean(filter(lambda y: y!=0, map(lambda x: pitch_op[x][i], pitch_op)))
#    var_op[i] = np.var(filter(lambda y: y!=0, map(lambda x: pitch_op[x][i], pitch_op)))
#    
#    i = u'Intensity(dB)'
#    moy_op[i] = np.mean(map(lambda x: intensity_op[x][i], intensity_op))
#    var_op[i] = np.var(map(lambda x: intensity_op[x][i], intensity_op))
    
    # scripts praat USER    
#    print "Lancement des scripts praat user..." # Creer une VARIABLE praat a base de fichiers .wav
#    
#    text_formants_us = pl.run_script('formants.praat', S_PATH+"normalized/us_"+name, 5, 5500)
#    formants_us = pl.read_praat_out(text_formants_us)
#    
#    text_pitch_us = pl.run_script('pitch.praat', S_PATH+"normalized/us_"+name)
#    pitch_us = pl.read_praat_out(text_pitch_us)
#    
#    text_intensity_us = pl.run_script('intensity.praat', S_PATH+"normalized/us_"+name)
#    intensity_us = pl.read_praat_out(text_intensity_us)
    
#    moy_us = {}
#    var_us = {}
#    for i in [u'B1', u'B2', u'F1', u'F2']:
#        moy_us[i] = np.mean(map(lambda x: formants_us[x][i], formants_us))
#        var_us[i] = np.var(map(lambda x: formants_us[x][i], formants_us))
#        
#    moy_us[u'Pitch'] = np.mean(filter(lambda y: y!=0, map(lambda x: pitch_us[x][u'Pitch'], pitch_us)))
#    var_us[u'Pitch'] = np.var(filter(lambda y: y!=0, map(lambda x: pitch_us[x][u'Pitch'], pitch_us)))
#    
#    moy_us[u'Intensity(dB)'] = np.mean(map(lambda x: intensity_us[x][u'Intensity(dB)'], intensity_us))
#    var_us[u'Intensity(dB)'] = np.var(map(lambda x: intensity_us[x][u'Intensity(dB)'], intensity_us))



############################ TOUR DE PAROLE : TEXTE ###########################

    # Pour fichiers txt mainteant afin de synchro les tours de paroles --> chargement des fichiers
    try:
        f_op = open(txt_op,'Ur')
        turns_op = f_op.read().split('\n.\n') # .split('\n.\n') : tour par tour de parole 
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
        print cpt_op
        print cpt_us
        # .split('\n')[1:] pour enlever le header et separer par utterance
        turn_op = turns_op[cpt_op].split('\n')[1:] 
        turn_us = turns_us[cpt_us].split('\n')[1:]
            
        try:
            start_op = float(turn_op[0].split()[0]) # temps du debut de parole
        except IndexError:
            start_op = float('Inf')
            
        try:
            start_us = float(turn_us[0].split()[0]) # temps du debut de parole
        except IndexError:
            start_us = float('Inf')
        try:
            if start_op < start_us: # Si il y a coupure dans la phrase d'un loc : il
                dump = read_turn(formants["op"], pitch["op"], intensity["op"], turn_op)
                cpt_op += 1
            else:
                dump = read_turn(formants["us"], pitch["us"], intensity["us"], turn_us)
                cpt_us += 1
        except BaseException, e:
            print e
            print turn_op
            print turn_us
        complete_dump += "\n\n"+dump
        if cpt_op >= len(turns_op)-1 and cpt_us >= len(turns_us)-1:
            break

    # avoir 3 digits
    if len(name) == 1:
        name2 = '00'+name
    elif len(name) == 2:
        name2 = '0'+name
    else:
        name2 = name
        
    f2 = open(D_PATH+'/dump_'+name2, 'w')
    f2.write(complete_dump)
    f2.close()
    
#read_file('25')

#%% read all files
# NORMALIZED afin de normaliser les fichiers audio du style
# 2009.01.06.14.53.49_Operator HeadMounted_Spike.wav --> ne peut pas etre utiliser sans audiolab
# du coup utiliser directement les fichiers normalisés
 
for session_dir in sorted(os.listdir(S_PATH)):
    if session_dir == "normalized":
        continue
    else:
        read_file(session_dir, NORMALIZED = False)