# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:00:31 2015

@author: Valou

Formatage audio

Le module qui pond les dump audio.
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
la main en inversant des tours de parole.
"""
#%%
#from scikits.audiolab import Sndfile, Format
#from praatinterface import PraatLoader
import numpy as np
import glob
import os
from sys import platform
from subprocess import Popen, PIPE
from time import time

#S_PATH = '/home/lucasclaude3/Documents/Stage_Telecom/Datasets/Semaine/Sessions/'
#D_PATH = '/home/lucasclaude3/Documents/Stage_Telecom/MonProjet/'

# To know if I am on the MAC or on the PC with Linux             
CURRENT_OS = platform   
if CURRENT_OS == 'darwin': # MAC       
    INIT_PATH = "/Users/Valou/"
    PRAAT_NAME = "Praat.app/Contents/MacOS/Praat"
elif CURRENT_OS == 'linux2': # LINUX
    INIT_PATH = "/home/valentin/"
    PRAAT_NAME = "praat_linux"

S_PATH = INIT_PATH + "Dropbox/TELECOM_PARISTECH/Stage_Lucas/Datasets/Semaine/Sessions/"
DUMP_PATH = INIT_PATH + "Dropbox/TELECOM_PARISTECH/Stage_Lucas/Datasets/Semaine/all/"
D_PATH = INIT_PATH + "Dropbox/TELECOM_PARISTECH/Stage_Lucas/MonProjet/"

#%% First step : preprocessing



#%% Second step : processing (Praat) --> formants+pitch+intensite

## Praat.app/Contents/MacOS/Praat --run praatScripts/formants.praat /Users/Valou/Dropbox/TELECOM_PARISTECH/Stage_Lucas/MonProjet/tests_audio/wavtest_normalized.wav 5 5500
# subprocess.call([D_PATH+"Praat.app/Contents/MacOS/Praat", "--run", D_PATH+"praatScripts/formants.praat", 
# D_PATH+"tests_audio/wavtest_normalized.wav", "5", "5500"])

#P_PATH  = "/Applications/Praat.app/Contents/MacOS/"

############ SINCE PRAATINTERFACE IS SHIT ###################

def praat_interface_val(var_name, file_name):
    """
    Use the subprocess.Popen function to call Praat using the shell and take the output
    Return a dict with variable[time][specific_type]
    """    
    
    praat_var = {}
    

    # If there is more arguments in the Praat script    
    if isinstance(var_name,tuple):
        arg_shell = [D_PATH + PRAAT_NAME, "--run", D_PATH+"praatScripts/"+var_name[0]+".praat", 
                 file_name]
        for i in range(1,len(var_name)):
            arg_shell.append(var_name[i])
        #script_name= var_name[0]
        script_name= var_name[0] +'_'+ file_name.split('/')[-1].split('.')[0]
    else:
        arg_shell = [D_PATH + PRAAT_NAME, "--run", D_PATH+"praatScripts/"+var_name+".praat", 
                 file_name]
        #script_name= var_name
        script_name= var_name +'_'+ file_name.split('/')[-1].split('.')[0]
                 
        
    #popen = Popen([D_PATH+"Praat.app/Contents/MacOS/Praat", "--run", D_PATH+"praatScripts/"+var_name+".praat", 
                              #D_PATH+"tests_audio/wavtest_normalized.wav", "5", "5500"],stdout = PIPE)
    print "lancement script " + arg_shell[2].split('/')[-1] + " sur " + script_name
    FILE_OUTPUT = True
    if FILE_OUTPUT:
        f = file(D_PATH+'praat_outputs/data_praat_'+script_name+'.txt','w+')
        debut = time()
        popen = Popen(arg_shell,stdout = f)
        popen.wait()
        fin = time()
        f.close()
        print "script " + arg_shell[2].split('/')[-1] + " sur " + file_name.split('/')[-1].split('.')[0] + " execute en %.2f mn" %((fin-debut)/60)
        return None
        
    debut = time()
    popen = Popen(arg_shell,stdout = PIPE)
    milieu = time()
    stdout, _ = popen.communicate()
    fin = time()
    print "script " + arg_shell[2].split('/')[-1] + " execute/communique en %.2f/%.2f mn" %((milieu-debut,fin-milieu)/60)
    # To separate per line
    stdout = stdout[:-1].split('\n')
    
    # time F1 B1 F2 B2 for the formants
    script_outputs = stdout[0].split('\t')[1:]
    # time length (number of lines)
    t_len = len(stdout)-1
    
    # per element in each line
    for x in range(t_len):
        # Outputs at time t ; t(1) = t_0
        std_t = stdout[x+1].split('\t')
        time_t = float(std_t[0])
        praat_var[time_t] = {}
        
        if not std_t[1] == '--undefined--':
            # per column 
            for elements in script_outputs:
                praat_var[time_t][elements] = float(std_t[script_outputs.index(elements)+1])
        else:
            # per column ; if there is no pitch for example
            for elements in script_outputs:
                praat_var[time_t][elements] = -1          
        
    
    return praat_var
#%% tests & co
"""
variables = [('formants','5','5500'),'pitch','intensity']
#variables = ['intensity']
praat_var = {}
file_name = D_PATH + 'tests_audio/wavtest_normalized.wav'
spkr = "op"
name = "25"
file_name = S_PATH+"normalized/"+spkr+"_"+name+".wav"
for var_name in variables:
    if isinstance(var_name,tuple):
        praat_var[var_name[0]] = praat_interface_val(var_name, file_name)
    else:
        praat_var[var_name] = praat_interface_val(var_name, file_name)
"""
##################### POUR JUSTE GARDER LES FICHIERS TEXT

variables = [('formants','5','5500'),'pitch','intensity']
praat_var = {}
names_all = ['114', '95', '126', '90', '26', '27', '37', '29', '83', '53', '127', '100', '43', '66']


FORMATAGE_ALL = False
if FORMATAGE_ALL:
    for spkr in ["op","us"]:
        for name in names_all:
            file_name = S_PATH+"normalized/"+spkr+"_"+name+".wav"
            for var_name in variables:
                praat_interface_val(var_name, file_name)
else: # juste le user 25 qui n'était pas fait
    spkr = "us"
    name = '25'
    file_name = S_PATH+"normalized/"+spkr+"_"+name+".wav"
    for var_name in variables:
        praat_interface_val(var_name, file_name)


#%% compute mean and var for preprocessing
"""
formants = praat_var['formants']
pitch = praat_var['pitch']
intensity = praat_var['intensity']
"""
#%% Third step : implement read_turn

def read_turn(formants, pitch, intensity, turn, forbidden): 
    """
    On lit les dates de debut et de fin de chaque mot, et on 
    prend moyenne et variance sur la durée correspondante.
    On effectue une normalisation Z pour chaque composante (X-mu)/sigma
    """
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
    """

    features = {}
    text = ""
    for line in turn:
        if line == '':
            continue
        
        words = line.split()
        start = float(words[0]) / 1000
        end = float(words[1]) / 1000
        """
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
        """
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
            #forbidden = 0 # since sometimes there is no pitch
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

#%% Dechet ancien
"""   
f = open(D_PATH + 'transcript_test', 'Ur') #Transcript used for a test : normally done in read_file 
text = f.read()
turns = text.split('\n.\n')
turn = turns[0]
turn_formated = turn.split('\n')[1:]
dump = read_turn(formants, pitch, intensity, turn_formated)
"""
#%% Fourth step : read a file

def read_file(session_dir, NORMALIZED_AUDIO = False):
    """
    Pour une seule session
    On calcule les features audio qui sont faits toutes les 20ms et et on les
    moyennes sur la durée des mots
    Les fichiers normalisés sont dans le folder normalized ss la forme :
    op_25.wav et us_25.wav
    """
    # All the differents scripts used 
    variables = [('formants','5','5500'),'pitch','intensity']
    
    # name is the number of the session
    name = session_dir 
    
    # Si on normalise les fichiers d'origine, sinon on prend ceux deja normalisés (+ simple)
    if NORMALIZED_AUDIO:
            pass
            
    # trouver les fichiers txt: -->
    l_op = glob.glob(S_PATH+name+'/word*operator*')
    l_us = glob.glob(S_PATH+name+'/word*user*')
    if len(l_op) != 1 or len(l_us) != 1:
        raise Exception('Zero ou multiples matchs pour les fichiers txt')
    else:
        txt_op = l_op[0]
        txt_us = l_us[0]
             
    ################################ AUDIO ####################################
   
    # Scripts praat OPERATOR
    print "Lancement des scripts praat operator, puis user" # Cree une VARIABLE praat a base de fichiers .wav

    formants = {}
    pitch = {}
    intensity = {}
    moy = {}
    var = {}
    
    for spkr in ["op", "us"]:
        moy[spkr] = {}
        var[spkr] = {}
        
        praat_var = {}
        # name of the wavfile used to extract the praat features
        wavfile_name = S_PATH+"normalized/"+spkr+"_"+name+".wav"
        #wavfile_name = D_PATH + 'tests_audio/wavtest_normalized.wav'
        for var_name in variables:
            if isinstance(var_name,tuple):
                praat_var[var_name[0]] = praat_interface_val(var_name,wavfile_name)
            else:
                praat_var[var_name] = praat_interface_val(var_name,wavfile_name)
        
        formants[spkr] = praat_var['formants']
        pitch[spkr] = praat_var['pitch']
        intensity[spkr] = praat_var['intensity']

        ####################        
        # Taking the mean and the var of each feature for each speaker 
        for i in [u'B1', u'B2', u'F1', u'F2']:
            moy[spkr][i] = np.mean(map(lambda x: formants[spkr][x][i], formants[spkr]))
            var[spkr][i] = np.var(map(lambda x: formants[spkr][x][i], formants[spkr]))            
            # Normalization
            for x in formants[spkr].keys():
                formants[spkr][x][i] = (formants[spkr][x][i] - moy[spkr][i])/np.sqrt(var[spkr][i])
        
        # y != 0 pour ne prendre que les moments ou l'on a du pitch
        # Pitch = -1 when undefined
        i = u'Pitch'
        moy[spkr][i] = np.mean(filter(lambda y: y!=-1, map(lambda x: pitch[spkr][x][i], pitch[spkr])))
        var[spkr][i] = np.var(filter(lambda y: y!=-1, map(lambda x: pitch[spkr][x][i], pitch[spkr])))
        # Normalization
        for x in pitch[spkr].keys():
            pitch[spkr][x][i] = (pitch[spkr][x][i] - moy[spkr][i])/np.sqrt(var[spkr][i])
        
        # Value that you'll find on the pitch when it's not defined
        no_pitch = (-1 - moy[spkr][i])/np.sqrt(var[spkr][i])
        
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
            if start_op < start_us: # If operator speaks before user
                dump = read_turn(formants["op"], pitch["op"], intensity["op"], turn_op, no_pitch)
                cpt_op += 1
            else:
                dump = read_turn(formants["us"], pitch["us"], intensity["us"], turn_us, no_pitch)
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
        
    f2 = open(DUMP_PATH+'dump_audio_val/dump_'+name2+'.txt', 'w+')
    f2.write(complete_dump[2:])
    f2.close()
    
#read_file('25', NORMALIZED_AUDIO = False)

#%% read all files
# NORMALIZED afin de normaliser les fichiers audio du style
# 2009.01.06.14.53.49_Operator HeadMounted_Spike.wav --> ne peut pas etre utiliser sans audiolab
# du coup utiliser directement les fichiers normalisés
LOOP = False
if LOOP: 
    for session_dir in sorted(os.listdir(S_PATH)):
        if session_dir == "normalized":
            continue
        else:
            read_file(session_dir, NORMALIZED_AUDIO = False)