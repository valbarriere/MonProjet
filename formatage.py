# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import nltk

def recPrint(t,i):
    """
        affiche l'arbre du XML de manière visible
    """
    
    print("\t"*i+"tag: "+str(t.tag))
    print("\t"*i+"attrib: "+str(t.attrib))
    print("\t"*i+"text: "+str(t.text))     
    for child in t:
        if str(child.tag) != "relation":
            recPrint(child,i+1)
    return

def acToFormat1(acfilename, aafilename):
    """
        convertit les annotations ac et aa de Glozz au format Conll
    """

    f_ac = open(acfilename, 'Ur')
    tree = ET.parse(aafilename)
    root = tree.getroot()
    idx_sentences = []
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
            
            # modifs a partir d'ici            
            
            for tagType in child.findall(".//*[type]"):
                nameType = tagType[0].text
                f_ac.seek(i0)
                annotation = f_ac.read(i1 - i0)
                str_sentence = nltk.word_tokenize(annotation)
                pos_sentence = nltk.pos_tag(str_sentence)
    
                if nameType == "paragraph":
                    i2 = i0+1+len(str_sentence[0])
                    cpt = 2
                    idx_sentence = []
                    while cpt<len(str_sentence):
                        i2+=1+len(str_sentence[cpt-1])
                        features_dict[i2] = pos_sentence[cpt]
                        labels_dict[i2] = "N"
                        idx_sentence.append(i2)
                        cpt += 1
                    
                    idx_sentences.append(idx_sentence)
                    
                else:
                    
                    ## disjonction source target appraisal
                    
                    if "source" in nameType:
                        attitudeType = "source"
                    elif "target" in nameType:
                        attitudeType = "target"
                    elif "Utterance" in nameType:
                        attitudeType = str(tagType[1][0].attrib)
                        
                    if attitudeType != "none":
                        i2=i0
                        cpt=0
                        while cpt<len(str_sentence):
                            while not labels_dict.has_key(i2):
                                i2=i2+1
                            if labels_dict[i2] == "N":
                                labels_dict[i2]=attitudeType
                            else:
                                labels_dict[i2]=labels_dict[i2]+";"+attitudeType
                                
                            i2+=len(str_sentence[cpt])
                            cpt+=1
    
    keys = sorted(labels_dict.keys(),reverse=False)
    
    currentWord_labels = labels_dict[keys[0]].split(";")
    if currentWord_labels != ["N"]:
        try:
            currentWord_labels[0]='B'+currentWord_labels[0]
        except IndexError:
            pass
        labels_dict[keys[0]]=';B'.join(currentWord_labels)
    
    k=1
    while k<len(keys):
        previousWord_labels = currentWord_labels
        currentWord_labels = labels_dict[keys[k]].split(";")
        if currentWord_labels == ["N"]:
            k+=1
            continue
        else:
            str_labels = ""
            for lab in currentWord_labels:
                if lab in previousWord_labels:
                    str_labels += "I"+lab+";"
                else:
                    str_labels += "B"+lab+";"
            labels_dict[keys[k]]=str_labels[:-1]
            k+=1
    
    f1_file = []
    for idx_sentence in idx_sentences:
        f1_sentence = []
        for idx in idx_sentence:
            f1_sentence.append(";".join(features_dict[idx])+";"+labels_dict[idx])
        f1_file.append(f1_sentence)
    
    return f1_file