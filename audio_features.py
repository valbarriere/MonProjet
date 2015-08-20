# -*- coding: utf-8 -*-
u"""Tests sur l'audio.

Attention : chargement des wav à faire en python2 !!!
warning alsa backend à résoudre mais pas gênant ici
"""

from features import mfcc
from scikits.audiolab import Sndfile, Format
import numpy as np
from praatinterface import PraatLoader

#%% lecture/écriture : ok

f = Sndfile("wavtest.wav")

n = int(f.nframes)
fs = f.samplerate
nc = f.channels
enc = f.encoding

data = f.read_frames(n, np.float32)
data_truncated = data[:100000,]

f_truncated = Sndfile("wavtest_truncated.wav", 'w', Format('wav', 'pcm24'), 1, 48000)
f_truncated.write_frames(data_truncated)
f_truncated.close()

#%% mfcc : ok

mfcc_feat = mfcc(data_truncated, fs)

#%% script praat pitch : ok

path = '/home/lucasclaude3/Documents/Stage_Telecom/MonProjet/'
pl = PraatLoader(praatpath=path+'praat')
text = pl.run_script('features.praat', path+'wavtest_truncated', 5, 5500)
# attention read_praat_out pas adapté au problème

#%% script praat mfcc : en cours

pl = PraatLoader(praatpath=path+'praat')
mfcc_features = pl.run_script('wav2mfcc.praat', path+'wavtest_truncated', 1.0, 2.0)