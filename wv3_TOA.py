#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import math
import numpy as np
from rios import applier
from rios import fileinfo
from rios import pixelgrid
from rios import cuiprogress


def createESunDict():
    #Thuillier corrections
    eSun = {'b1Esun': 1757.89,
          'b2Esun': 2004.61,
          'b3Esun': 1830.18,
          'b4Esun': 1712.07,
          'b5Esun': 1535.33,
          'b6Esun': 1348.08,
          'b7Esun': 1055.94,
          'b8Esun': 858.77}
    return eSun          


def getParams(metadataFile):
    f = open(metadataFile, 'r')
    calFactors =[]
    layerNames = []
    band = 0
    for line in f:
       words = line.strip().split("=")
       if len(words)>1:
           label = words[0].strip()
           dataVal = words[1].strip()
           if label == "BEGIN_GROUP":
               bandName = dataVal.strip(';')
               if len(layerNames) < 8:
                    print(band)
                    layerNames.append(bandName)
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

    solarZenithAngle = np.radians(90.0-meanSunEl)
    eSun = createESunDict()
    return calFactors, solarZenithAngle, eSun, layerNames
    
    
def doTOA(info, inputs, outputs, otherargs):
    """
    Called by RIOS. 
    
    Create natural colour image
    
    """
    toaStack = []
    #raw = otherargs.inscale.DNtoUnits(inputs.raw)
    for i, band in enumerate(inputs.raw):
        #print(band)
        ESunName = "b%sEsun"%str(i+1)
        eSunBand = float(otherargs.eSun[ESunName])
        #((3.14159265359*1.0336585561*(float(b8)*1.065140e-02;/8.890000e-02;))/(cos(0.816814089933)*858.77))
        #toaBand = ((otherargs.pi*(float(otherargs.d)**2)*band*otherargs.calFactors[i][2]/otherargs.calFactors[i][3])/(cos(otherargs.solarZenithAngle)*eSunBand))
        toaBand = (otherargs.pi*float(otherargs.d)**2*band*otherargs.calFactors[i][2]/otherargs.calFactors[i][3])/(math.cos(otherargs.solarZenithAngle)*eSunBand)
        toaStack.append(toaBand)
    colourStack = np.array(toaStack)

    outputs.toaFile = colourStack.astype(np.float32)
    #nullMask = (inputs.ref[0] == otherargs.refNull)
    #for i in range(3):
       # outputs.naturalcolour[i][nullMask] = otherargs.outNull    
    
     

def main():
    metadataFile = sys.argv[1]
    imgFile = sys.argv[2]
    toaFile = sys.argv[3]
    d = float(sys.argv[4])
    outfile = "%s_envi.exp"%(metadataFile.split('.')[0])
    
    if os.path.exists(toaFile):
        os.remove(toaFile)
    
    (calFactors, solarZenithAngle, eSun, layerNames) = getParams(metadataFile)  
        
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherargs = applier.OtherInputs()
    controls = applier.ApplierControls()
    
    infiles.raw = imgFile
    outfiles.toaFile = toaFile
    
    otherargs.calFactors = calFactors
    otherargs.solarZenithAngle = solarZenithAngle
    otherargs.pi = 3.14159265358979323846
    otherargs.d = d
    otherargs.eSun = eSun    

    info = fileinfo.ImageInfo(imgFile)
    xMin = np.floor(info.xMin)
    xMax = np.ceil(info.xMax)
    yMin = np.floor(info.yMin)
    yMax = np.ceil(info.yMax)
    proj = info.projection
    transform = info.transform
    xRes = info.xRes
    yRes = info.yRes
    otherargs.outNull = 0
    print(otherargs.outNull)
    
    controls.setStatsIgnore(otherargs.outNull)
    controls.setOutputDriverName('GTiff')
    controls.setCreationOptions(['COMPRESS=LZW'])
    controls.setCreationOptions(['BIGTIFF=IF_SAFER'])

    
    pixgrid = pixelgrid.PixelGridDefn(geotransform=transform, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xRes=xRes, yRes=yRes,
        projection=proj)
    print(pixgrid)    
    controls.setReferencePixgrid(pixgrid)    
    controls.setLayerNames(layerNames)
    controls.setWindowXsize(20)
    controls.setWindowYsize(20)
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    
    applier.apply(doTOA, infiles, outfiles, otherargs, controls=controls)    


if __name__ == "__main__":
    main()
