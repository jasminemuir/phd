#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy
import numpy.ma
#import qvf
from pylidar import lidarprocessor
from rios import cuiprogress
from rios import pixelgrid
from pylidar.toolbox import interpolation
#from matplotlib import mlab
#from rsc.utils import history

def index_at(idx, shape, axis=-1):
    if axis<0:
        axis += len(shape)
    shape = shape[:axis] + shape[axis+1:]
    index = list(numpy.ix_(*[numpy.arange(n) for n in shape]))
    index.insert(axis, idx)
    return tuple(index)

    
def writeImageFunc(data, otherargs):

    pts = data.input1.getPointsByBins(colNames=['Z','X','Y'])
    #print("points",pts.shape)
    pxlCoords = data.info.getBlockCoordArrays()
    
    out = numpy.empty((3, pxlCoords[0].shape[0], pxlCoords[0].shape[1]), dtype=numpy.float64)
    nullval = otherargs.ignore
    
    zVals = pts['Z']
    xVals = pts['X']
    yVals = pts['Y']

    if pts.shape[0] > 0:
        minZ = numpy.around(zVals.min(axis=0), decimals=2)
        idx = numpy.argmin(zVals,axis=0)
        x = numpy.around(xVals[index_at(idx, xVals.shape, axis=0)], decimals=2)
        y = numpy.around(yVals[index_at(idx, yVals.shape, axis=0)], decimals=2)
        #print(minZ,"minz\n")
        #print(x,"x\n")
#        x = xVals[~newMask][idx]
#        y = yVals[~newMask][idx]
#        
#        #x = numpy.ma.array(xMask[index_at(idx, xMask.shape, axis=0)], mask = newMask, dtype=numpy.float64)
#        #y = numpy.ma.array(yMask[index_at(idx, yMask.shape, axis=0)], mask = newMask, dtype=numpy.float64)
#
#        #x = x.flatten()
#        #y = y.flatten()
#        #minZ = minZ.flatten()
#        print(idx.shape,x.shape, y.shape, minZ.shape)
#        #out = interpolation.interpGrid(x, y, minZ, pxlCoords, otherargs.interp)
#        #out = numpy.expand_dims(out, axis=0).astype(numpy.float64)
#        #overshootMask = numpy.isnan(out)
#        #out[overshootMask] = otherargs.ignore  
        out = numpy.ma.array((minZ,x,y))
        #print(out, "out\n")
#  
#        #print(out.shape)
    else:
        out.fill(otherargs.ignore)    
    data.imageOut1.setData(out)
      
    
    
def testWrite(infile, outfile, binSize):
    outNull = 0     
    
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(infile, lidarprocessor.READ)
    dataFiles.imageOut1 = lidarprocessor.ImageFile(outfile, lidarprocessor.CREATE)
    dataFiles.imageOut1.setRasterIgnore(outNull)

    controls = lidarprocessor.Controls()
    #controls.setOverlap(5)
    #controls.setWindowSize(30)
    #controls.setReferenceResolution(binSize)
    controls.setWindowSize(16)
    progress = cuiprogress.GDALProgressBar()
    controls.setProgress(progress)
    
    otherargs = lidarprocessor.OtherArgs()
    otherargs.ignore = outNull
    otherargs.interp = "pynn"
    otherargs.minVal = None
    otherargs.outNull = outNull
    
    lidarprocessor.doProcessing(writeImageFunc, dataFiles, otherArgs=otherargs, controls=controls)
    
if __name__ == '__main__':
    testWrite(sys.argv[1], sys.argv[2],0.1)
