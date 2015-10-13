# -*- coding: utf-8 -*-
u"""Méthodes pour formater les données brutes (txt et xml)."""

import xml.etree.ElementTree as ET
import nltk
import os

path = "/home/lucasclaude3/Documents/Stage_Telecom/Datasets/Semaine/all/"


""" En gros, à l'origine on a un fichier XML pas beau duquel on veut extraire
les informations qui nous intéressent, i.e. les "units" correspondant aux
attitudes, sources et targets.

Dur dur de faire un code plus joli mais ca marche... Si tu ne comprends pas le
principe des annotations n'hesite pas a demander à Caroline, c'est elle qui les
a faite, et elle ne mord (presque) pas ! """

def __attitude(nameType, tagType):
    """ c'est juste pour automatiser certains trucs chiants."""
    att = "none"
    if "source" in nameType:
        att = "source"
    elif "target" in nameType:
        att = "target"
    elif "Utterance" in nameType:
        att = tagType[1][0].text
        pol = tagType[1][2].text
        if att != "none" and pol != "undefined":
            att = "%s_%s" % (att, pol)
        else:
            att = "none"
    return att


def __updatelabelBIO(lab, attitudeType, cpt):
    """ a l'origine il n'y a pas de B et de I, il faut les creer soit même."""
    if lab == "O":
        if cpt == 0:
            labB = 'B-' + attitudeType
        else:
            labB = 'I-' + attitudeType
    else:
        if cpt == 0:
            labB = lab + ";B-" + attitudeType
        else:
            labB = lab + ";I-" + attitudeType
    return labB


def dump_semaine(ac_filename, aa_filename, dump_filename):
    u"""convertit les annotations ac et aa de Glozz au format Conll."""
    f_ac = open(ac_filename, 'Ur')
    tree = ET.parse(aa_filename)
    root = tree.getroot()
    idx_sentences = []
    features_dict = {}
    labels_dict = {}
    for child in root:
        if child.tag == "unit":
            for startChar in child.iter("start"): # trouver le début du "unit"
                index = startChar[0]
                i0 = int(index.attrib["index"])
            for endChar in child.iter("end"): # trouver la fin
                index = endChar[0]
                i1 = int(index.attrib["index"])
            for tagType in child.findall(".//*[type]"):
                nameType = tagType[0].text
                f_ac.seek(i0)
                annotation = f_ac.read(i1 - i0)
                str_sentence = nltk.word_tokenize(annotation)
                pos_sentence = nltk.pos_tag(str_sentence)
                if nameType == "paragraph": # le texte proprement dit
                    i2 = i0 + 1 + len(str_sentence[0])
                    cpt = 2
                    idx_sentence = []
                    while cpt < len(str_sentence): # parcourir les mots et les stocker
                        i2 += 1 + len(str_sentence[cpt-1])
                        features_dict[i2] = pos_sentence[cpt]
                        labels_dict[i2] = "O"
                        idx_sentence.append(i2)
                        cpt += 1
                    idx_sentences.append(idx_sentence)
                else:
                    attitudeType = __attitude(nameType, tagType) # les labels !
                    if attitudeType != "none":
                        i2 = i0
                        cpt = 0
                        while cpt < len(str_sentence):
                            while i2 not in labels_dict:
                                i2 = i2 + 1
                            label_alone = labels_dict[i2]
                            labels_dict[i2] = __updatelabelBIO(label_alone,
                                                               attitudeType,
                                                               cpt) 
                            # on vient de rajouter les labels découverts dans le dico de chaque mot
                            i2 += len(str_sentence[cpt])
                            cpt += 1
    f_ac.close()

    f = open(dump_filename, 'w')
    for idx_sentence in idx_sentences:
        for idx in idx_sentence:
            f.write("\t".join(features_dict[idx])+"\t"+labels_dict[idx])
            f.write('\n')
        f.write('\n\n')
    f.close()

dump_semaine(path+"/ac1/session025.ac", path+"/aa1/session025.aa", "dump_025")

#%% dump all files

for filename in os.listdir(path+"/ac1"):
    dump_semaine(path+"/ac1/"+filename, path+"/aa1/"+filename[:-3]+".aa",
                 path+"/dump/dump_"+filename[-6:-3])
