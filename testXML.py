u"""juste pour visualiser le XML."""
# -*- coding: utf-8 -*-

from __future__ import division

import xml.etree.ElementTree as ET

"""Permet d'afficher le XML des annotations pour mieux comprendre.
Ouvre fichier xml de type .aa --> fichier GLOZZ = annotations de Caroline 
"""

#path = "/home/lucasclaude3/Documents/Stage_Telecom/Datasets/Semaine/all/aa1/session025.aa"
path = "/Users/Valou/Documents/TELECOM_PARISTECH/Stage_Lucas/Datasets/Semaine/all/aa1/session025.aa"

tree = ET.parse(path)
root = tree.getroot()

def recPrint(t, i, max_iter):
    u"""affiche le XML avec des tabulations.
    A chaque fois que i augmente, une tabulation de plus
    Pour voir les tags de Caroline    
    """
    print("\t"*i+"tag: "+str(t.tag))
    print("\t"*i+"attrib: "+str(t.attrib))
    print("\t"*i+"text: "+str(t.text))
    cpt = 0
    for child in t:
        cpt += 1
        if cpt == max_iter: # pour ne pas tout afficher et voir un peu ce qui spasse
            return
        if str(child.tag) != "relation" and str(child.tag) != "schema":
            recPrint(child, i+1, max_iter)
    return

recPrint(root, 0, 50)
