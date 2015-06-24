# -*- coding: utf-8 -*-

from __future__ import division
from nltk.corpus import wordnet as wn

import os
import nltk
import xml.etree.ElementTree as ET
import pycrfsuite
import numpy as np

path = "C:/Users/claude-lagoutte/Documents/LUCAS/These/Python/Datasets/Semaine"
tree = ET.parse(path+"/train/aa1/session025.aa")
root = tree.getroot()

def recPrint(t,i):
    print("\t"*i+"tag: "+str(t.tag))
    print("\t"*i+"attrib: "+str(t.attrib))
    print("\t"*i+"text: "+str(t.text))     
    for child in t:
        if str(child.tag) != "relation" and str(child.tag) != "schema":
            recPrint(child,i+1)
    return

recPrint(root,0)