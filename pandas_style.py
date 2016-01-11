# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:40:15 2015

@author: valentin

Transform the dump.txt of praat in dump pandas
"""

import pandas as pd
import numpy as np
from sys import platform

# To know if I am on the MAC or on the PC with Linux             
CURRENT_OS = platform   
if CURRENT_OS == 'darwin': # MAC       
    INIT_PATH = "/Users/Valou/"
elif CURRENT_OS == 'linux2': # LINUX
    INIT_PATH = "/home/valentin/"

DATA_PATH = INIT_PATH + 'Dropbox/TELECOM_PARISTECH/Stage_Lucas/MonProjet/praat_outputs/'

datas = {}
var_tot = ['formants', 'intensity', 'pitch']
spkrs = ['op', 'us']
sessions =  ['25', '114', '95', '126', '90', '26', '27', '37', '29', '83', '53', '127', '100', '43', '66']

session = sessions[0]
var_name = var_tot[0]
spkr = spkrs[0]

def change_time(df):
    df = datas[var_tot[0]]['time']
    # nombre de valeurs qui se suivent dans le temps
    np.sum([df[i] == df[i+1]-0.001 for i in range(len(df)-1)])
    bool1 = [np.abs(df[i+1]- (df[i]+0.001)) <0.00001 for i in range(len(df)-1)]
    bool2 = [np.abs(df[i+1]- df[i]) <0.00001 for i in range(len(df)-1)]
    df2 = datas[var_tot[1]]['time']
    
    # put the time as index
    time_name = datas[var_name].keys()[0]
    datas[var_name] = datas[var_name].set_index(time_name)
    
    return  None


for var_name in var_tot:
    # import the file
    datas[var_name] = pd.read_table(DATA_PATH + 'data_praat_'+var_name+'_'+ spkr +'_'+ session +'.txt', sep = '\t')
    
    datas[var_name] = change_time(datas[var_name])

    
    #  concatenation
    total = datas[var_tot[0]]
    for i in range(1,len(var_tot)):
        total = total.join(datas[var_tot[i]])
#%%    
for var_name in var_tot:
    # import the file
    datas[var_name] = pd.read_table(DATA_PATH + 'data_praat_'+var_name+'_'+ spkr +'_'+ session +'.txt', sep = '\t')
#%%

df = datas[var_tot[0]]['time']
# nombre de valeurs qui se suivent dans le temps
np.sum([df[i] == df[i+1]-0.001 for i in range(len(df)-1)])
bool1 = [np.abs(df[i+1]- (df[i]+0.001)) <0.00001 for i in range(len(df)-1)]
bool2 = [np.abs(df[i+1]- df[i]) <0.00001 for i in range(len(df)-1)]
df2 = datas[var_tot[1]]['time']
    

#%%

var_name = var_tot[0]
time_name = datas[var_name].keys()[0]
df = datas[var_name][time_name]

z = []
eps = 1e-4
for i in range(len(df)-1):
    if np.abs(df[i+1]- df[i]) < 0.001 - eps: # si moins decart que 0.001
        z.append(i)
print "longueur moins d'ecart que 1e-3 : %d" %(len(z))

q = []
for i in range(len(df)-1):
    if np.abs(df[i+1]- df[i]) > 0.001 + eps: # si plus decart que 0.001
        #q.append(df['time'][i+1]- df['time'][i])
        q.append(i)
print "longueur moins d'ecart que 1e-3 : %d" %(len(q))  

dist_max = None
for k in range(10,0,-1):
    if np.sum([z[i] -q[i] <= k for i in range(len(z))]) == len(z):
        dist_max = k

print "distance maximum d'écart entre 1 blanc et un saut : %d" %(dist_max)

l = []
eps = 1e-4
for i in range(len(df)-1):
    if np.abs(df[i+1]- df[i]) > 0.001 + eps: # si plus decart que 0.001
        l.append(df[i+1]- df[i])
print "ecart plus grand que 2e-3 : %d" %(max(l) > 0.002 + eps)
#%% 
"""generalement le code sort 2 fois le meme temps t et fianlement il saute un 
autre temps t un peu plus loin (au pire dist_max)
z[count[]] donne l'indice temporel ou cela s'arrete, le saut suivant qui retablie est 
a z[count[]]+ dist_max
"""

count = []
for i in range (len(z)):    
    if z[i] - q[i] == dist_max:
        count.append(i)


#%% verification ecart pas plus grand que 0.002
l = []
eps = 1e-4
for i in range(len(df)-1):
    if np.abs(df[i+1]- df[i]) > 0.001 + eps: # si plus decart que 0.001
        l.append(df[i+1]- df[i])
print 'ecart plus grand que 2e-3 : '
max(l) > 0.002 + eps
  

#%% 
"""nombre d'écart = 1 ou 0 (sachant que 0 est nul car c'est des indices et il ne peut pas y avoir le meme indice)
"""
np.sum([z[i] -q[i] < 2  for i in range(len(z))])
     
#%%
k=6
dfv = df[10*k:10*(k+1)]
dfv2 = df2[10*k:10*(k+1)]
try:
    dfv3 = pd.concat([dfv,dfv2], axis=1)
    dfv.join(dfv2)
    
except Exception,e:
    print e, k
#%%
k=64
dfv = df[1*k:1*(k+1)]
dfv2 = df2[1*k:1*(k+1)]
try:
    dfv3 = pd.concat([dfv,dfv2], axis=1)
except Exception,e:
    print e, k        