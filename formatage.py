# -*- coding: utf-8 -*-
u"""méthodes pour formater les données brutes (txt et xml)."""

import xml.etree.ElementTree as ET
import nltk
import os


def recPrint(t, i):
    u"""affiche l'arbre du XML de manière visible."""
    print("\t"*i+"tag: "+str(t.tag))
    print("\t"*i+"attrib: "+str(t.attrib))
    print("\t"*i+"text: "+str(t.text))
    for child in t:
        if str(child.tag) != "relation":
            recPrint(child, i+1)
    return


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
            for startChar in child.iter("start"):
                index = startChar[0]
                i0 = int(index.attrib["index"])
            for endChar in child.iter("end"):
                index = endChar[0]
                i1 = int(index.attrib["index"])

            for tagType in child.findall(".//*[type]"):
                nameType = tagType[0].text
                f_ac.seek(i0)
                annotation = f_ac.read(i1 - i0)
                str_sentence = nltk.word_tokenize(annotation)
                pos_sentence = nltk.pos_tag(str_sentence)
                if nameType == "paragraph":
                    i2 = i0 + 1 + len(str_sentence[0])
                    cpt = 2
                    idx_sentence = []
                    while cpt < len(str_sentence):
                        i2 += 1 + len(str_sentence[cpt-1])
                        features_dict[i2] = pos_sentence[cpt]
                        labels_dict[i2] = "O"
                        idx_sentence.append(i2)
                        cpt += 1
                    idx_sentences.append(idx_sentence)
                else:
                    if "source" in nameType:
                        attitudeType = "source"
                    elif "target" in nameType:
                        attitudeType = "target"
                    elif "Utterance" in nameType:
                        attitudeType = str(tagType[1][0].text)
                    if attitudeType != "none":
                        i2 = i0
                        cpt = 0
                        while cpt < len(str_sentence):
                            while i2 not in labels_dict:
                                i2 = i2 + 1
                            if labels_dict[i2] == "O":
                                if cpt == 0:
                                    labels_dict[i2] = 'B-' + attitudeType
                                else:
                                    labels_dict[i2] = 'I-' + attitudeType
                            else:
                                if cpt == 0:
                                    labels_dict[i2] = labels_dict[i2] + ";B-"\
                                        + attitudeType
                                else:
                                    labels_dict[i2] = labels_dict[i2] + ";I-"\
                                        + attitudeType
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


def dump_datasetsemaine(path):
    u"""Créer les fichiers dump au format Conll pour le dataset SEMAINE.

    Path doit être le chemin absolu du dossier contenant les deux sous-dossiers
    avec les fichiers aa et ac
    """
    for filename in os.listdir(path+"/ac1"):
        dump_semaine(path+"/ac1/"+filename, path+"/aa1/"+filename[:-3]+".aa",
                     path + "/dump/dump_" + filename[:-3])
