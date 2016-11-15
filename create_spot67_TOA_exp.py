#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import math
import numpy as np
from datetime import date
from xml.dom import minidom

metadataFile = sys.argv[1]
d = sys.argv[2]
outfile = "%s_envi.exp"%(metadataFile.split('.')[0])

if os.path.exists(outfile):
    os.remove(outfile)

#Open output envi expression file for writting    
o = open(outfile,'w')
o.write("ENVI EXPRESSIONS\n")    

#Open and read xml file
dimXml = open(metadataFile,'r').read()
dimapdoc = minidom.parseString(dimXml)

#Find sun zenith
sunElevation = dimapdoc.getElementsByTagName("SUN_ELEVATION")[0].childNodes[0].data
sunZenith = np.radians(90.0 - float(sunElevation))

#Find L=DN/GAIN+BIAS formula

bandIDNodes = dimapdoc.getElementsByTagName("BAND_ID")
gainNodes = dimapdoc.getElementsByTagName("GAIN")
biasNodes = dimapdoc.getElementsByTagName("BIAS")
solarIrrNodes = dimapdoc.getElementsByTagName("VALUE") 

#Iterate through the node lists and build a list of calibration factors
numbands = 4 ## SPOT 6/7 have 4 bands

calFacList=[]
for i in range(numbands):
    bandID = bandIDNodes[i].childNodes[0].data
    bandIDImage = "B%d" %(int(bandID[-1])+1)    #Convert Band name to same as image bands to avoid confusion
    gain = gainNodes[i].childNodes[0].data
    bias = biasNodes[i+4].childNodes[0].data
    solarIrr = solarIrrNodes[i].childNodes[0].data 
    calFacList.append([bandIDImage,gain,bias,solarIrr])
    
#sort the calFactList
calFacList.sort()


pi = "3.14159265358979323846"       
 
#Create output expression for each band and write to envi expression file
for band in calFacList:
    expressionbase = "((%s*%s*((float(%s)/%s)+%s))/(cos(%s)*%s))"%(pi,float(d)**2,band[0],band[1],band[2],sunZenith,band[3])
    print(expressionbase)
    o.write("%s\n"%expressionbase)
    
o.close()
