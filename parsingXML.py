# -*- coding: utf-8 -*-

from __future__ import division
from nltk.corpus import wordnet as wn

import os
import nltk
import xml.etree.ElementTree as ET
import pycrfsuite
import numpy as np

path = "C:/Users/claude-lagoutte/Documents/LUCAS/These/Python/Datasets/Semaine"
tree = ET.parse(path+"/aa1/session025.aa")
root = tree.getroot()

#%% Obtention des features et des labels
            
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    synset = sent[i][2]
    talker = sent[i][3]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        'synset=' + synset,
        'talker=' + talker
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


#%% Pour comprendre la structure du XML

for child in root :
    print("Nouvelle annotation : " + str(child.tag))
    print(child.attrib)
    
    for s in child :
        print("a) Et voici un nouveau noeud : " + str(s.tag)) 
        for t in s :
            print("\t b) Et voila un sous-noeud : " + str(t.tag))
            print("\t    " + str(t.attrib))
            print("\t    " + str(t.text))
            for u in t :
                print("\t \t c) Et voila un sousou : " + str(u.tag))
                print("\t \t    " + str(u.attrib))
                print("\t \t    " + str(u.text))
                for v in u :
                    print("\t \t \t d) Et enfin un saoul : " + str(v.tag))
                    print("\t \t \t    " + str(v.attrib))
                    print("\t \t \t    " + str(v.text))
                    
#%% Extraire les infos d une annotation

child = root[1] ## paragraph
        
for startChar in child.iter("start"):
    index = startChar[0]
    i0 = int(index.attrib["index"])
for endChar in child.iter("end"):
    index = endChar[0]
    i1 = int(index.attrib["index"])

f = open(path+"/ac1/session025.ac",'Ur')
f.seek(i0)
schema = f.read(i1-i0)
print(schema)

#%% Extraire toutes les annotations

morphy_tag={'NN':wn.NOUN,'JJ':wn.ADJ,'VB':wn.VERB,'RB':wn.ADV}
X=[]
y=[]
str_label = "appraisalItem"
str_labelB = "B"+str_label
str_labelI = "I"+str_label

for filename in os.listdir(path+"/ac1"):
    print(filename)
    f_ac = open(path+"/ac1/"+filename,'Ur')
    tree = ET.parse(path+"/aa1/"+filename[:-2]+"aa")
    root = tree.getroot()
    sentences = []
    features_dict = {}
    labels_dict = {}
    for child in root :
        if child.tag == "unit":
            for startChar in child.iter("start"):
                index = startChar[0]
                i0 = int(index.attrib["index"])
                
            for endChar in child.iter("end"):
                index = endChar[0]
                i1 = int(index.attrib["index"])
            
            for tagType in child.iter("type"):
                nameType = tagType.text
                f_ac.seek(i0)
                annotation = f_ac.read(i1-i0)
                str_sentence = nltk.word_tokenize(annotation)
                pos_list = nltk.pos_tag(str_sentence)
    
                if nameType == "paragraph":
                    if "user" in str_sentence[1]:
                        participant = "user"
                    elif "operator" in str_sentence[1]:
                        participant = "operator"
                    i2 = i0+1+len(str_sentence[0])
                    cpt=2
                    sentence=[]
                    while cpt<len(str_sentence):
                        i2+=1+len(str_sentence[cpt-1])
                        u=pos_list[cpt]                    
                        try:
                            v=morphy_tag[u[1][:2]]
                            w=wn.synsets(u[0],pos=v)
                        except KeyError:
                            w=wn.synsets(u[0])
                            
                        if len(w)>0:
                            h_root=w[0].root_hypernyms()
                        else:
                            h_root=[]
                        
                        features_dict[i2]=u[0]+";"+u[1]+";"+str(h_root)+";"+participant
                        labels_dict[i2]="N"
                        sentence.append(i2)
                        cpt+=1
                    
                    sentences.append(sentence)
                    
                else:
                    i2=i0
                    cpt=0
                    while cpt<len(str_sentence):
                        while not labels_dict.has_key(i2):
                            i2=i2+1
                        if labels_dict[i2] == "N":
                            labels_dict[i2]=nameType
                        else:
                            labels_dict[i2]=labels_dict[i2]+";"+nameType
                            
                        i2+=len(str_sentence[cpt])
                        cpt+=1
    
    keys = sorted(features_dict.keys(),reverse=False)
    
    currentWord = labels_dict[keys[0]].split(";")
    if currentWord != ["N"]:
        try:
            currentWord[0]='B'+currentWord[0]
        except IndexError:
            pass
        labels_dict[keys[0]]=';B'.join(currentWord)
    
    k=1
    while k<len(keys):
        previousWord = currentWord
        currentWord = labels_dict[keys[k]].split(";")
        if currentWord == ["N"]:
            k+=1
            continue
        else:
            list_lab = []
            for lab in currentWord:
                if lab in previousWord:
                    list_lab.append("I"+lab)
                else:
                    list_lab.append("B"+lab)
            labels_dict[keys[k]]=list_lab
            k+=1

    for sentence in sentences:
        sent=[]
        sent_labels=[]
        cpt=0
        for index in sentence:
            sent.append(features_dict[index].split(";"))
            if str_labelB in str(labels_dict[index]) :
                sent_labels.append(str_labelB)
            elif str_labelI in str(labels_dict[index]) :
                sent_labels.append(str_labelI)
            else:
                sent_labels.append("N")
            cpt+=1
        X.append(sent2features(sent))
        y.append(sent_labels)
    f_ac.close()

#%% CRFsuite

trainer1 = pycrfsuite.Trainer(verbose=False)
trainer2 = pycrfsuite.Trainer(verbose=False)
Xtest1=[]
ytest1=[]
Xtest2=[]
ytest2=[]
for xseq, yseq in zip(X, y):
    r = np.random.random_sample()
    if r>0.2:
        trainer1.append(xseq,yseq)
        Xtest2.append(xseq)
        ytest2.append(yseq)
    else:
        trainer2.append(xseq,yseq)
        Xtest1.append(xseq)
        ytest1.append(yseq)

trainer1.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer2.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer1.train('model1')
trainer2.train('model2')

tagger1 = pycrfsuite.Tagger()
tagger1.open('model1')

nb_tokens = 0
nb_truepos = 0
nb_falsepos = 0
nb = 0

cpt = 0
while cpt < len(ytest1) :
    phrase = Xtest1[cpt]
    corr = ytest1[cpt]
    pred = tagger1.tag(phrase)
        
    i=0
    while i<len(phrase):
        if str_label in corr[i]:
            nb+=1
            if str_label in pred[i]:
                nb_truepos+=1
        
        if str_label not in corr[i]:
            if str_label in pred[i]:
                nb_falsepos+=1
        i+=1
    
    nb_tokens+=len(phrase)
    cpt+=1

precision = nb_truepos/(nb_truepos+nb_falsepos)
recall = nb_truepos/nb
print("Precision ",str_label," : ", precision)
print("Recall: ",str_label," : ", recall)