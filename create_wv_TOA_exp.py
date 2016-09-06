#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import math
import numpy as np

metadataFile = sys.argv[1]
d = sys.argv[2]
outfile = "%s_envi.exp"%(metadataFile.split('.')[0])

if os.path.exists(outfile):
    os.remove(outfile)

f = open(metadataFile, 'r')
o = open(outfile,'w')
o.write("ENVI EXPRESSIONS\n")

calFactors =[]
#meanSunE1
#absCalFactor
#effectiveBandwidth
band = 0
for line in f:
    words = line.strip().split("=")
    if len(words)>1:
        label = words[0].strip()
        dataVal = words[1].strip()
        if label == "BEGIN_GROUP":
            bandName = dataVal.strip(';')
        if label == "absCalFactor":
            absCalFactor = float(dataVal.strip(';'))           
        elif label == "effectiveBandwidth":
            effectiveBandwidth = float(dataVal.strip(';'))
            band += 1
            bandNo = "b%s" %band
            calFactors.append([bandName,bandNo, absCalFactor, effectiveBandwidth])
        elif label == "meanSunEl":
            meanSunEl = float(dataVal.strip(';'))    
       
    
f.close()

print(calFactors)
print(meanSunEl)

solarZenithAngle = np.radians(90.0-meanSunEl)

#Thuillier corrections
eSun = {'b1Esun': 1757.89,
       'b2Esun': 2004.61,
       'b3Esun': 1830.18,
       'b4Esun': 1712.07,
       'b5Esun': 1535.33,
       'b6Esun': 1348.08,
       'b7Esun': 1055.94,
       'b8Esun': 858.77}
       
print(eSun)
pi = "3.14159265358979323846"       
 
for band in calFactors:
    ESunName = "%sEsun"%band[1]
    print(band[0],band[1])
    expressionbase = "((%s*%s*(float(%s)*%s/%s))/(cos(%s)*%s))"%(pi,float(d)**2,band[1],band[2],band[3],solarZenithAngle,eSun[ESunName])
    print(expressionbase)
    #o.write("%s: %s\n"%(band[0],band[1]))
    o.write("%s\n"%expressionbase)
    
o.close()



     
