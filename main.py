# -*- coding: utf-8 -*-

import formatage
import extraction
import cPickle as pickle

path = "C:/Users/claude-lagoutte/Documents/LUCAS/These/Python/Datasets/Semaine"
d1 = formatage.acToFormat1(path+"/ac1/session025.ac",path+"/aa1/session025.aa")
dump1 = open('dumpsession025','w')
pickle.dump(d1,dump1)
dump1.close()

d2 = formatage.format1ToFormat2(d1,["N","BoperatorUtterance","IoperatorUtterance"])
f = extraction.toFeatures(d2)
l = extraction.toLabels(d2,[1, 3])